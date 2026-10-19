#!/usr/bin/env python3
"""使用 StarVLA 现有训练组件的单卡训练入口。

ACT 和 DP 由两个独立的单进程、单物理 GPU 作业运行。本入口沿用现有优化器、
checkpoint 和 GT 评估循环，只记录单卡启动审计，不启用或宣称 DDP。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


WORKSPACE = Path(__file__).resolve().parents[2]
STARVLA = WORKSPACE / "starvla_local"
sys.path.insert(0, str(STARVLA))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training import train_starvla as upstream
from starVLA.training.trainer_utils.config_tracker import wrap_config
from launch_protocol import audit_prelaunch_directory, write_failure_preserving


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class SingleGPUVLATrainer(upstream.VLATrainer):
    """沿用上游训练循环，并使用单 rank 启动审计约定。"""

    def _train_step(self, batch_vla, batch_vlm=None):
        del batch_vlm
        framework = self.config.framework
        chunk = int(framework.get("chunk_size", framework.get("horizon", 16)))
        action_dim = int(framework.get("action_dim", 20))
        state_dim = int(framework.get("state_dim", 20))
        image_keys = list(framework.get("image_keys", []))
        expected_batch = int(self.config.datasets.vla_data.per_device_batch_size)
        actual_batch = len(batch_vla)
        # 数据加载器在每轮末尾可能返回较小的剩余 batch。
        # 配置的 batch 大小作为上限，同时允许有效的尾批次。
        require_full_batch = bool(self.config.datasets.vla_data.get("drop_last", False))
        if require_full_batch and actual_batch != expected_batch:
            raise RuntimeError(
                f"single-GPU full batch contract failed: got {actual_batch}, expected {expected_batch}"
            )
        if not 0 < actual_batch <= expected_batch:
            raise RuntimeError(
                f"single-GPU batch size out of range: got {actual_batch}, expected 1..{expected_batch}"
            )
        first = batch_vla[0]
        action_shape = tuple(np.asarray(first["action"]).shape)
        state_shape = tuple(np.asarray(first["state"]).reshape(-1).shape)
        image_shapes = [tuple(np.asarray(image).shape) for image in first["image"]]
        if action_shape != (chunk, action_dim):
            raise RuntimeError(f"action contract failed: {action_shape} != {(chunk, action_dim)}")
        if state_shape != (state_dim,):
            raise RuntimeError(f"state contract failed: {state_shape} != {(state_dim,)}")
        if len(first["image"]) != len(image_keys) or image_shapes != [(224, 224, 3)] * len(image_keys):
            raise RuntimeError(f"camera contract failed: {image_shapes} for {image_keys}")

        from train_image_augmentation import augment_training_batch
        batch_vla = augment_training_batch(batch_vla, self.config.get("h20_augmentation_contract", None))

        probe_parameter = None
        probe_before = None
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with _autocast_context(self.accelerator.device):
                output_dict = self.model.forward(batch_vla, training_step=self.completed_steps)
                action_loss = output_dict["action_loss"]
                if not torch.isfinite(action_loss).all():
                    raise RuntimeError(f"non-finite action loss at step {self.completed_steps}: {action_loss}")
                total_loss = action_loss
            self.accelerator.backward(total_loss)

            gradient_probe = None
            if self.completed_steps < 2:
                for parameter in self.model.parameters():
                    if parameter.requires_grad and parameter.grad is not None and parameter.grad.detach().float().norm().item() > 0:
                        # 在反向传播后选择参数更新见证，避免审计选中一个虽有效、
                        # 但没有被模型首个 loss 实际更新的参数。
                        probe_parameter = parameter
                        probe_before = parameter.detach().float().clone()
                        gradient_probe = parameter.grad.detach().float()
                        break
                if gradient_probe is None or probe_parameter is None or probe_before is None:
                    raise RuntimeError("no trainable gradient found during single-GPU startup audit")
                grad_norm = gradient_probe.norm()
                grad_checksum = gradient_probe.sum()

            clip_value = self.config.trainer.gradient_clipping
            if clip_value is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        step_number = self.completed_steps + 1
        if self.completed_steps < 2:
            parameter_delta = (probe_parameter.detach().float() - probe_before).norm()
            memory_mb = float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
            reserved_memory_mb = float(torch.cuda.max_memory_reserved() / 2**20) if torch.cuda.is_available() else 0.0
            audit = {
                "optimizer_step": step_number,
                "process_count": int(self.accelerator.num_processes),
                "batch_size": len(batch_vla),
                "item_shapes": {
                    "action": [chunk, action_dim],
                    "state": [state_dim],
                    "cameras": [len(image_keys), 224, 224, 3],
                },
                "gradient_norms": [float(grad_norm.item())],
                "gradient_checksums": [float(grad_checksum.item())],
                "gradient_checksum_spread": 0.0,
                "parameter_update_norms": [float(parameter_delta.item())],
                "cuda_peak_memory_mb": [memory_mb],
                "cuda_peak_memory_reserved_mb": [reserved_memory_mb],
            }
            self._ddp_audit_records.append(audit)
            if self.accelerator.is_main_process:
                output_dir = Path(self.config.output_dir)
                (output_dir / "single_gpu_startup_audit.json").write_text(
                    json.dumps(self._ddp_audit_records, indent=2) + "\n"
                )
                if step_number == 2:
                    if not (parameter_delta.item() > 0 and grad_norm.item() > 0):
                        raise RuntimeError(f"single-GPU startup verification failed: {audit}")
                    receipt = {
                        "status": "PASS",
                        "verified_optimizer_steps": 2,
                        "launch_mode": "single_process_single_gpu",
                        "process_count": 1,
                        "batch_size": len(batch_vla),
                        "gradient_synchronization": "not applicable; no distributed reduction",
                        "records": self._ddp_audit_records,
                        "created_unix_time": time.time(),
                        "initial_checkpoint": getattr(self, "_initial_checkpoint_contract", None),
                    }
                    (output_dir / "startup_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

        if self.accelerator.is_main_process and step_number == 1:
            atomic_json(
                Path(self.config.output_dir) / "first_loss.json",
                {
                    "status": "PASS",
                    "optimizer_step": 1,
                    "loss": float(action_loss.detach().float().item()),
                    "model": str(self.config.framework.name),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "local_cuda_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1,
                    "pid": os.getpid(),
                    "unix_time": time.time(),
                },
            )

        metrics = {
            "action_loss": float(action_loss.detach().float().item()),
            "action_dit_loss": float(action_loss.detach().float().item()),
        }
        action_metrics = output_dict.get("action_metrics", {})
        if isinstance(action_metrics, dict):
            for name, value in action_metrics.items():
                if value is not None:
                    if torch.is_tensor(value):
                        value = value.detach().float().mean().item()
                    metrics[f"action/{name}"] = float(value)
        if self.accelerator.is_main_process:
            record = {"optimizer_step": step_number, "unix_time": time.time(), **metrics}
            with (Path(self.config.output_dir) / "optimizer_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return metrics


def run(config_path: Path, overrides: list[str] | None = None) -> None:
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = str(config_path.resolve())
    cfg = wrap_config(cfg)

    # Formal launch is deliberately narrow: this copied entrypoint cannot be
    # pointed at any other split or mixed historical configuration.
    cfg_text = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
    for forbidden in ("train149", "train133", "val6", "1004", "multigrasp"):
        if forbidden in cfg_text:
            raise RuntimeError(f"forbidden historical token in formal config: {forbidden}")
    active_retry_run_id = str(cfg.run_id).startswith(("ACT_1008_only_delta_10k_coexist_", "ACT_1008_only_delta_10k_coexist_smoke_", "ACT_1008_h20_aug_h16_20k_retry", "DP_1008_h20_aug_h16_20k_retry"))
    active_1015_run_id = str(cfg.run_id).startswith("DP_1015_h20_aug_h16_")
    if str(cfg.run_id) not in {
        "ACT_1008_h20_aug_h16_smoke_v4",
        "DP_1008_h20_aug_h16_smoke_v2",
        "ACT_1008_h20_aug_h16_10k",
        "DP_1008_h20_aug_h16_10k",
        "ACT_1008_h20_aug_h16_20k",
        "DP_1008_h20_aug_h16_20k",
        "DP_1008_h20_aug_h16_40k",
        "ACT_1008_h20_aug_h16_smoke",
        "ACT_1008_h20_aug_h16_smoke_v2",
        "ACT_1008_h20_aug_h16_smoke_v3",
        "DP_1008_h20_aug_h16_smoke",
        "ACT_1008_only_delta_10k_20261009",
        "DP_1008_only_delta_10k_20261009",
        "ACT_1008_only_delta_10k_smoke_20261009",
        "DP_1008_only_delta_10k_smoke_20261009",
        "ACT_1008_only_delta_10k_coexist_20261010",
        "ACT_1008_only_delta_10k_coexist_smoke_20261010",
    } and not active_retry_run_id and not active_1015_run_id:
        raise RuntimeError(f"unexpected 1008-only run_id: {cfg.run_id}")
    allowed_roots = {
        (WORKSPACE.parent / "runs").resolve(),
        (WORKSPACE / "runs/1008_10k").resolve(),
        (WORKSPACE / "runs/preflight").resolve(),
        (WORKSPACE / "runs/a100_coexist").resolve(),
        Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/runs").resolve(),
    }
    if Path(str(cfg.run_root_dir)).resolve() not in allowed_roots:
        raise RuntimeError(f"unexpected formal run root: {cfg.run_root_dir}")
    _gt_manifest = Path(str(cfg.trainer.get("gt_manifest", ""))).resolve()
    _allowed_gt = {
        (WORKSPACE / "dataset_trimmed_letterbox224_v1/training_manifest.json").resolve(),
        Path("/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/data/training_split_1015_letterbox224_v1/training_manifest.json").resolve(),
    }
    if _gt_manifest not in _allowed_gt:
        raise RuntimeError("formal GT manifest is not a recognized training manifest")
    manifest = json.loads(Path(str(cfg.trainer.gt_manifest)).read_text())
    if manifest.get("status") != "PASS":
        raise RuntimeError("formal GT manifest status is not PASS")
    _train_eps = int(manifest.get("train_episodes", -1))
    _val_eps = int(manifest.get("val_episodes", -1))
    if active_1015_run_id:
        if _train_eps != 36 or _val_eps != 4:
            raise RuntimeError(f"1015 manifest must be 36 train/4 val, got {_train_eps}/{_val_eps}")
    elif _train_eps != 17 or _val_eps != 1:
        raise RuntimeError("formal GT manifest is not PASS with 17 train/1 val")

    _expected_procs = 2 if active_1015_run_id else 1
    if upstream.accelerator.num_processes != _expected_procs:
        raise RuntimeError(
            f"process count contract failed: got {upstream.accelerator.num_processes}, expected {_expected_procs}"
        )
    continuation_checkpoint = cfg.trainer.get("continuation_checkpoint", None)
    continuation_start_step = int(cfg.trainer.get("continuation_start_step", 0))
    if cfg.trainer.get("pretrained_checkpoint", None) not in (None, "", False):
        raise RuntimeError("formal train17 runs must start from an empty pretrained checkpoint")
    if continuation_checkpoint and not (0 < continuation_start_step < int(cfg.trainer.max_train_steps)):
        raise RuntimeError("continuation checkpoint requires start step within (0, max_train_steps)")
    _allowed_mix = {"g1_pick_apple_1008_train17_c16", "g1_pick_apple_1015_train36_c16"}
    if str(cfg.datasets.vla_data.data_mix) not in _allowed_mix:
        raise RuntimeError(f"unexpected data mix: {cfg.datasets.vla_data.data_mix}")
    if int(cfg.datasets.vla_data.per_device_batch_size) != 32:
        raise RuntimeError("formal train17 runs require batch size 32")

    output_dir = Path(str(cfg.run_root_dir)) / str(cfg.run_id)
    # 启动器写 command.txt、容量审计和 formal 显存复核后才调用本入口。
    # 统一协议只放行这些精确启动审计；checkpoint/failure/旧模型和任何
    # 训练生成物都仍然是硬拒绝，且不会被清理或覆盖。
    # 双卡时 rank0 会先写审计产物，审核只在主进程做，之后同步再继续。
    if upstream.accelerator.is_main_process:
        audit_prelaunch_directory(output_dir)
    upstream.accelerator.wait_for_everyone()
    output_dir = upstream.setup_directories(cfg)
    if upstream.accelerator.is_main_process:
        shutil.copyfile(config_path, output_dir / "launch_config.yaml")
        atomic_json(
            output_dir / "run_contract.json",
            {
                "data_root": str(Path(cfg.datasets.vla_data.data_root_dir).resolve()),
                "source_data_root": str(
                    Path(cfg.datasets.vla_data.data_root_dir).resolve()
                    / "train17"
                ),
                "train_split": "train17",
                "validation_split": "val1",
                "manifest": str(cfg.trainer.get("gt_manifest", "")),
                "framework": str(cfg.framework.name),
                "chunk_size": int(cfg.framework.get("chunk_size", cfg.framework.get("horizon", 16))),
                "action_dim": int(cfg.framework.action_dim),
                "state_dim": int(cfg.framework.state_dim),
                "image_keys": list(cfg.framework.image_keys),
                "from_scratch": not bool(cfg.trainer.get("pretrained_checkpoint", None)),
                "continuation_checkpoint": str(continuation_checkpoint) if continuation_checkpoint else None,
                "continuation_start_step": continuation_start_step,
            },
        )

    model = upstream.build_framework(cfg)
    continuation_contract = None
    if continuation_checkpoint:
        checkpoint_path = Path(str(continuation_checkpoint)).resolve()
        if not checkpoint_path.is_file():
            raise RuntimeError(f"continuation checkpoint missing: {checkpoint_path}")
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        incompatible = model.load_state_dict(checkpoint, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict continuation load unexpectedly reported: {incompatible}")
        continuation_contract = {
            "status": "PASS", "resume_kind": "model_weights_only_reconstruction",
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": digest,
            "strict_model_load": True, "starting_optimizer_step": continuation_start_step,
            "optimizer_state_loaded": False, "scheduler_state_loaded": False,
            "optimizer_and_scheduler": "rebuilt; scheduler advanced deterministically to continuation_start_step",
        }
        if upstream.accelerator.is_main_process:
            atomic_json(output_dir / "continuation_receipt.json", continuation_contract)
            atomic_json(output_dir / "status.json", {"phase": "continuation_startup", "latest_step": continuation_start_step, "last_gt_completed": None, "checkpoint": str(checkpoint_path), "errors": [], "continuation": continuation_contract})
    if upstream.accelerator.is_main_process:
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        atomic_json(
            output_dir / "model_parameter_counts.json",
            {
                "status": "PASS",
                "framework": str(cfg.framework.name),
                "total_parameters": int(total_parameters),
                "trainable_parameters": int(trainable_parameters),
                "from_scratch": not bool(cfg.trainer.get("pretrained_checkpoint", None)),
                "optimizer_steps_target": int(cfg.trainer.max_train_steps),
                "created_unix_time": time.time(),
            },
        )
    # 训练 loader 仅依据 cfg.datasets.vla_data.data_mix 构建；train17 是唯一
    # 训练数据。val1 只作为独立 GT 评估输入，不会传入 prepare_data。
    dataloader = upstream.prepare_data(cfg, upstream.accelerator, output_dir)
    optimizer, scheduler = upstream.setup_optimizer_and_scheduler(model, cfg)
    trainer = SingleGPUVLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        accelerator=upstream.accelerator,
    )
    # 训练只使用上面构建的训练加载器；验证集由独立 GT 评估流程读取。
    trainer.prepare_training()
    if continuation_contract:
        trainer.completed_steps = continuation_start_step
        trainer._initial_checkpoint_contract = continuation_contract
        trainer._adjust_lr_scheduler_for_resume()
        if upstream.accelerator.is_main_process:
            atomic_json(output_dir / "startup_receipt.json", {"status": "PASS", "launch_mode": "single_process_single_gpu", "process_count": 1, "batch_size": int(cfg.datasets.vla_data.per_device_batch_size), "continuation": continuation_contract, "created_unix_time": time.time()})
    trainer.train()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    try:
        run(args.config.resolve(), overrides)
    except Exception as exc:
        cfg = OmegaConf.load(args.config)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        error_dir = Path(str(cfg.run_root_dir)) / str(cfg.run_id)
        error_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "pid": os.getpid(),
            "unix_time": time.time(),
        }
        write_failure_preserving(error_dir, failure)
        raise


if __name__ == "__main__":
    main()
