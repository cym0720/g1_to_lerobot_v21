#!/usr/bin/env python3
"""CPU-only source/config/window/concatenation audit; never train or adapt data.

Uses existing 15000-step GT arrays, exact AST-extracted indexing/queue methods,
and read-only saved configs/checkpoint buffers. Only --output is written.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import os
import sys
from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import json
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.normalize import Normalize, Unnormalize

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path("/home/cym/Projects_2026/starvla-g1-validation")
REPO = ROOT / "third_party/starvla"
NATIVE = ROOT / ".venv/lib/python3.10/site-packages/lerobot"
RUN = Path("/home/cym/artifacts/starvla/runs/g1_act_0923_pick_apple_plate_16step")
DATASET = Path("/home/cym/datasets/g1/0923_g1_pick_apple/empty_text_task_1")
GT = WORKSPACE / "output/act_gt_audit_20260929/model_gt_15000"
NATIVE_RUN = WORKSPACE / "output/lerobot_act_control_20260929/native_train"
FILES = {
    "saved_config": RUN / "config.full.yaml",
    "star_act": REPO / "starVLA/model/framework/VM4A/ACT.py",
    "star_loader": REPO / "starVLA/dataloader/gr00t_lerobot/datasets.py",
    "star_normalizer": REPO / "starVLA/dataloader/gr00t_lerobot/transform/state_action.py",
    "star_dataset_builder": REPO / "starVLA/dataloader/lerobot_datasets.py",
    "star_dataloader": REPO / "starVLA/dataloader/__init__.py",
    "star_trainer": REPO / "starVLA/training/train_starvla.py",
    "star_optimizer": REPO / "starVLA/training/trainer_utils/trainer_tools.py",
    "star_wrapper": REPO / "deployment/model_server/policy_wrapper.py",
    "native_act": NATIVE / "policies/act/modeling_act.py",
    "native_act_config": NATIVE / "policies/act/configuration_act.py",
    "native_policy_config": NATIVE / "configs/policies.py",
    "native_dataset_config": NATIVE / "configs/default.py",
    "native_train_config": NATIVE / "configs/train.py",
    "native_trainer": NATIVE / "scripts/train.py",
    "native_dataset_factory": NATIVE / "datasets/factory.py",
    "native_dataset": NATIVE / "datasets/lerobot_dataset.py",
    "native_utils": NATIVE / "datasets/utils.py",
    "native_normalizer": NATIVE / "policies/normalize.py",
    "native_optimizer_factory": NATIVE / "optim/factory.py",
    "native_optimizer": NATIVE / "optim/optimizers.py",
    "gt_generator": WORKSPACE / "tools/testing/audit_g1_act_gt.py",
    "native_launcher": WORKSPACE / "tools/testing/train_g1_native_lerobot.py",
}


def extract(path, class_name, name, namespace=None):
    namespace = dict(np=np, torch=torch, Image=Image) if namespace is None else namespace
    tree = ast.parse(path.read_text())
    nodes = tree.body if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


def constructor_configs(cfg):
    namespace = dict(ACTConfig=ACTConfig, FeatureType=FeatureType, PolicyFeature=PolicyFeature,
                     OBS_STATE="observation.state", ACTION="action",
                     ACTDefaultConfig=SimpleNamespace(image_size=(224, 224), image_keys=("cam0_rgb", "cam1_rgb")),
                     framework_config=cfg.framework)
    for name in ["_image_feature_name", "_get_image_size", "_build_input_features", "_build_output_features"]:
        namespace[name] = extract(FILES["star_act"], None, name, namespace)
    tree = ast.parse(FILES["star_act"].read_text())
    init = next(n for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "ACT" for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "act_config" for t in n.targets))
    original = eval(compile(ast.Expression(assignment.value), str(FILES["star_act"]), "eval"), namespace)
    native = ACTConfig(device="cpu")  # config only, no policy or pretrained weights
    matched = ACTConfig(device="cpu", chunk_size=16, n_action_steps=16)
    keys = ["chunk_size", "n_action_steps", "n_obs_steps", "vision_backbone", "pretrained_backbone_weights", "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers", "n_decoder_layers", "use_vae", "latent_dim", "n_vae_encoder_layers", "dropout", "kl_weight", "pre_norm", "optimizer_lr", "optimizer_lr_backbone", "optimizer_weight_decay", "use_amp"]
    return dict(saved_star_constructor={k: getattr(original, k) for k in keys}, native_defaults={k: getattr(native, k) for k in keys}, native_horizon_matched={k: getattr(matched, k) for k in keys})


def sentinel_and_windows():
    n, h, dim = 37, 16, 8
    pred = np.arange(n)[:, None, None] * 1000 + np.arange(h)[None, :, None] * 1000 + np.arange(20)[None, None, :]
    tree = ast.parse(FILES["gt_generator"].read_text())
    plot = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plot_episode")
    assignment = next(n for n in ast.walk(plot) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "executed" for t in n.targets))
    expression = compile(ast.Expression(assignment.value), str(FILES["gt_generator"]), "eval")
    executed = eval(expression, dict(np=np, pred=pred, n=n, h=h, dim=dim))
    np.testing.assert_array_equal(executed, np.arange(n) * 1000 + dim)
    select = extract(FILES["native_act"], "ACTPolicy", "select_action")
    queue = SimpleNamespace(config=SimpleNamespace(temporal_ensemble_coeff=None, n_action_steps=h),
                            eval=lambda: None, _action_queue=deque(maxlen=h),
                            predict_action_chunk=lambda batch: torch.tensor(pred[batch["frame"]]).unsqueeze(0))
    queued = torch.stack([select(queue, {"frame": i}) for i in range(n)])[:, 0, dim].numpy()
    np.testing.assert_array_equal(queued, executed)
    # Exact wrapper method: confirm all steps survive the output denormalization.
    raw = pred[:2].astype(np.float32)
    stats_mean, stats_std = np.arange(20, dtype=np.float32), np.linspace(0.1, 1.0, 20, dtype=np.float32)
    wrapper = SimpleNamespace(_default_unnorm_key="test", _available_unnorm_keys=["test"],
                              _get_processor=lambda _: SimpleNamespace(apply_state=lambda s: s, unapply_actions=lambda a: a * stats_std + stats_mean),
                              _framework=SimpleNamespace(predict_action=lambda **_: {"normalized_actions": raw}))
    predict = extract(FILES["star_wrapper"], "PolicyServerWrapper", "predict_action")
    denorm = predict(wrapper, [{"state": np.zeros(20)}, {"state": np.zeros(20)}])["actions"]
    np.testing.assert_array_equal(denorm, raw * stats_std + stats_mean)
    native_query = extract(FILES["native_dataset"], "LeRobotDataset", "_get_query_indices")
    old_query = extract(FILES["star_loader"], "LeRobotSingleDataset", "get_state_or_action")
    old_pad = extract(FILES["star_loader"], "LeRobotSingleDataset", "retrieve_data_and_pad")
    records = [json.loads(line) for line in (DATASET / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    lengths = [r["length"] for r in records]
    boundaries = np.concatenate([[0], np.cumsum(lengths)])
    meta = json.loads((DATASET / "meta/modality.json").read_text())
    results = []
    for ep in [0, 45, 90]:
        df = pd.read_parquet(DATASET / f"data/chunk-000/episode_{ep:06d}.parquet")
        action = np.stack(df["action.qpos"])
        receiver = SimpleNamespace(delta_indices={"action.robot": np.arange(16)}, trajectory_lengths=[len(df)], get_trajectory_index=lambda _: 0,
                                   curr_traj_data=df, lerobot_modality_meta=SimpleNamespace(action={"robot": SimpleNamespace(**meta["action"]["robot"])}),
                                   metadata=SimpleNamespace(modalities=SimpleNamespace(action={"robot": SimpleNamespace(absolute=True)})))
        receiver.retrieve_data_and_pad = MethodType(old_pad, receiver)
        native_receiver = SimpleNamespace(episode_data_index={"from": torch.tensor(boundaries[:-1]), "to": torch.tensor(boundaries[1:])}, delta_indices={"action": list(range(16))})
        for start in [0, 52, min(104, len(df) - 16), len(df) - 1]:
            old = old_query(receiver, ep, "action", "action.robot", start)
            idx, pad = native_query(native_receiver, int(boundaries[ep] + start), ep)
            local = np.array(idx["action"]) - boundaries[ep]
            native_window = action[local]
            np.testing.assert_array_equal(old, native_window)
            np.testing.assert_array_equal(local, np.minimum(start + np.arange(16), len(df) - 1))
            results.append(dict(episode=ep, start=start, source_indices=local.tolist(), exact_equal=True, native_padding_true_count=int(pad["action_is_pad"].sum())))
    return dict(plot_concat_sentinel_exact=True, native_queue_sentinel_exact=True, wrapper_preserves_full_horizon=True,
                tested_shape=list(raw.shape), tested_offsets=list(range(16)), off_by_16_found=False,
                note="Exact plot concatenate expression, exact native queue method and exact Star wrapper method executed on time-encoded CPU sentinels. Native window comparison uses a canonical key in memory only; no dataset adapted.", real_windows=results)


def real_outputs_and_normalization(stats):
    norm_forward = extract(FILES["star_normalizer"], "Normalizer", "forward")
    norm_inverse = extract(FILES["star_normalizer"], "Normalizer", "inverse")
    receiver = SimpleNamespace(mode="mean_std", statistics={k: torch.tensor(v) for k, v in stats["action"].items()})
    feature = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))}
    modes = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    native_stats = {"action": {"mean": torch.tensor(stats["action"]["mean"]), "std": torch.tensor(stats["action"]["std"])}}
    native_norm, native_inverse = Normalize(feature, modes, native_stats), Unnormalize(feature, modes, native_stats)
    report = {}
    for ep in [0, 45, 90]:
        path = GT / f"episode_{ep:06d}_predictions.npz"
        with np.load(path) as data:
            gt = data["gt_action"]
            df = pd.read_parquet(DATASET / f"data/chunk-000/episode_{ep:06d}.parquet")
            np.testing.assert_array_equal(gt, np.stack(df["action.qpos"]))
            np.testing.assert_array_equal(data["state"], np.stack(df["observations.state.qpos"]))
            tensor = torch.from_numpy(gt)
            normalized = norm_forward(receiver, tensor)
            np.testing.assert_allclose(norm_inverse(receiver, normalized).numpy(), gt, atol=3e-7, rtol=0)
            native_normalized = native_norm({"action": tensor})["action"]
            native_roundtrip = native_inverse({"action": native_normalized})["action"]
            nonzero = np.array(stats["action"]["std"]) != 0
            difference = float((native_normalized[:, nonzero] - normalized[:, nonzero]).abs().max())
            quantized = normalized.half().float()
            episode = dict(npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), gt_and_state_exactly_match_training_parquet=True,
                           normalization_nonzero_std_native_vs_star_max_error=difference,
                           star_float16_target_max_error_normalized=float((normalized - quantized).abs().max()),
                           native_roundtrip_max_error_raw=float((native_roundtrip - tensor).abs().max()), modes={})
            n = len(gt)
            for mode in ["fp32", "autocast_bf16", "weights_bf16"]:
                pred = data[mode]
                assert pred.shape == (n, 16, 20)
                starts = np.arange(0, n - 16 + 1, 16)
                span = np.ptp(pred[starts, :, 8:15], axis=1)
                gt_index = starts[:, None] + np.arange(16)[None, :]
                concat = np.concatenate([pred[s, :min(16, n - s)] for s in range(0, n, 16)])
                repeated_zero = np.concatenate([np.repeat(pred[s, :1], min(16, n - s), axis=0) for s in range(0, n, 16)])
                shift_errors = {}
                source = np.arange(16, n - 16)
                for shift in [-16, 0, 16]:
                    shift_errors[str(shift)] = float(np.abs(pred[source, 0, 8:15] - gt[source + shift, 8:15]).mean())
                episode["modes"][mode] = dict(shape=list(pred.shape), median_chunk_span_right_arm= np.median(span, axis=0).tolist(),
                                                median_gt_chunk_span_right_arm=np.median(np.ptp(gt[gt_index][:, :, 8:15], axis=1), axis=0).tolist(),
                                                adjacent_whole_right_arm_repeat_fraction=float(np.mean(np.all(np.diff(pred[starts, :, 8:15], axis=1) == 0, axis=2))),
                                                concat_exactly_equals_repeat_step0=bool(np.array_equal(concat, repeated_zero)),
                                                concat_vs_repeat0_max_difference_raw=float(np.abs(concat[:, 8:15] - repeated_zero[:, 8:15]).max()),
                                                concat_right_arm_mae=float(np.abs(concat[:, 8:15] - gt[:, 8:15]).mean()),
                                                same_rows_step0_mae_for_gt_offset=shift_errors)
            report[str(ep)] = episode
    return report


def checkpoint_buffers(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    visual = {k: v for k, v in state.items() if "normalize_inputs" in k and "images" in k}
    bn = {k: v for k, v in state.items() if "model.backbone" in k and k.endswith(("running_mean", "running_var"))}
    return dict(checkpoint=str(path), parameter_dtypes=sorted(set(str(v.dtype) for v in state.values())),
                image_normalizer_buffers={k: dict(shape=list(v.shape), min=float(v.min()), max=float(v.max())) for k, v in visual.items()},
                frozen_bn_buffer_count=len(bn), all_running_mean_zero=all(bool((v == 0).all()) for k, v in bn.items() if k.endswith("running_mean")),
                all_running_var_one=all(bool((v == 1).all()) for k, v in bn.items() if k.endswith("running_var")))


def comparison_rows():
    return [
        ("时间窗口", "obs/state t；action t..t+15", "默认100步；显式chunk/n_action_steps=16后同样t..t+15", "同定义，不是±16错误", "star_loader:1713-1745; native_act_config:176-182; native_dataset:646-659"),
        ("输出消费", "wrapper返回[B,16,20]；现有GT全chunk拼接", "select_action队列依次消费0..15，不只消费0", "已通过sentinel；近常数来自保存的chunk", "star_wrapper:207-214; gt_generator:207-225; native_act:125-133"),
        ("视觉初始化", "ResNet18，pretrained_backbone_weights=None", "ResNet18_Weights.IMAGENET1K_V1", "独立变量；不是证明关闭预训练必定失败", "saved_config:21; star_act:104-114; native_act_config:108-110"),
        ("FrozenBN", "FrozenBatchNorm2d；checkpoint均值0/方差1", "同样FrozenBatchNorm2d；加载ImageNet已有buffers", "BN类型相同，初始化来源不同；不能当成两个不同架构", "native_act:337-343"),
        ("图像归一化", "RGB/255；ACT视觉mean0/std1(identity)", "use_imagenet_stats=True；(RGB/255-ImageNet mean)/std", "独立变量，必须与视觉初始化分别报告", "star_act:117-134,165-175; native_dataset_config:38; native_dataset_factory:30-33,113-116"),
        ("图像几何", "三路letterbox224黑边；PIL uint8 bilinear", "本次hook同样letterbox224黑边；float bilinear antialias", "几何已匹配；插值/uint8舍入不是逐像素相同；原生默认本身不含letterbox", "star_act:52-68; native_launcher:103-110,145-146"),
        ("动作/state归一化", "外部mean_std一次；内部identity；样本转float16", "policy内部mean_std一次，std+1e-8；通常FP32输入", "已验证非零std维数值近似相同；零std策略不同但不可盲目归因", "star_normalizer:137-153,202-205; star_loader:1393-1416; native_normalizer:167-172"),
        ("LR分组", "整个action_model含backbone初始1e-4；base2.5e-5并非backboneLR", "默认其余1e-5/backbone1e-5，显式两组", "独立变量；旧run所有可训练组件均在action_model组", "saved_config:44-46; star_optimizer:127-146; native_act:82-101; native_act_config:136-138"),
        ("AdamW/裁梯度", "betas(.9,.95),wd1e-8,eps1e-8,clip1", "ACT preset betas(.9,.999),wd1e-4,eps1e-8,clip10", "独立变量；不要把原生默认optimizer当成与旧run一致", "saved_config:57-66; native_act_config:163-167; native_optimizer:78-88"),
        ("LR/训练预算/恢复", "500stepwarmup+cosine到1e-6；15000step；恢复weights但无Adam状态", "本次15000step/batch8/freshpolicy；ACT preset无scheduler", "预算/batch匹配；scheduler、seed42/1000、原run恢复轨迹仍不同", "saved_config:3,40-49,67; star_trainer:229-288; native_act_config:169-170; native_launcher:139-144,173-195"),
        ("AMP/KL", "trainer强制CUDA BF16；KL直接BF16计算", "默认use_amp=False，FP32；KL表达式本身同样未强制FP32", "已确认旧KL舍入问题；原生若打开AMP不能宣称自动修复", "star_trainer:420-423; star_act:250-251; native_policy_config:64; native_trainer:70-73; native_act:169-173"),
        ("重建loss", "right arm/gripper权重8，其余.1，并除有效权重总和", "均匀20维masked L1.mean()，无右臂额外权重", "总loss不能横向直接比大小；需相同右臂raw MAE/span", "star_act:241-251; native_act:159-173"),
        ("末帧padding", "重复末行，但action_is_pad全False", "重复末行并对越界位置标action_is_pad=True", "确认差异；pad占目标cell约2.50%，不足以解释内部全坍缩", "star_loader:1559-1565; star_act:217-219; native_dataset:646-659"),
        ("采样", "默认balance_trajectory_weights=False；episode等权再均匀选帧，有放回", "DataLoader shuffle=True；一轮各帧遍历，无放回", "独立变量；不是所有episode同样frame权重", "star_dataloader:42-46; star_loader:2241-2260,2325-2341; native_trainer:165-185"),
        ("decoder/VAE", "同一LeRobot ACT核心：4encoder/1decoder/4VAE，零query内容、可学习位置query", "同一实现与默认配置；use_vae=True,KLweight10", "架构没有根本不同，native本身不会换成query-as-content", "star_act:104-115; native_act_config:117-133; native_act:513-522,655-683"),
        ("原始字段名", "observations.state.qpos/action.qpos由modality映射", "本次已canonical保值重命名为observation.state/action", "resolver只认exact action；本次兼容前提已满足；本审计未改数据", "native_dataset_factory:55-61; native_utils:442-469; native_launcher:139-167"),
    ]


def actual_native_launch():
    requested = json.loads((NATIVE_RUN / "requested_train_config.json").read_text())
    image = json.loads((NATIVE_RUN / "image_preprocessing.json").read_text())
    manifest = json.loads((NATIVE_RUN / "launch_manifest.json").read_text())
    source = FILES["native_launcher"].read_text()
    assert "result = old_update(*a, **kw)" in source
    assert "old_save(checkpoint_dir, step, cfg, policy, optimizer, scheduler)" in source
    assert "native.make_dataset = lambda current_cfg: dataset" in source
    return dict(requested_config=requested, image_preprocessing=image, launch_manifest=manifest,
                hooks_call_original_update_and_save=True,
                note="Actual launch config/source inspected, not imported/executed. make_dataset returns native LeRobotDataset with Letterbox; update/save hooks add only metrics/GT. Smoke model discarded; official train creates a fresh policy. Requested config precedes cfg.validate; source ACT preset resolves AdamW1e-5/wd1e-4/betas(.9,.999)/clip10, scheduler=None. No subsequent native GT used.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output/lerobot_act_control_20260929/config_audit")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = OmegaConf.load(FILES["saved_config"])
    stats = json.loads((RUN / "dataset_statistics.json").read_text())["new_embodiment"]
    existing = json.loads((GT / "report.json").read_text())
    print("Native config/source/window/concatenation audit; CUDA disabled; no model inference/training.", flush=True)
    report = dict(scope="Read-only configuration audit. CPU only. No model inference, optimizer, dataset adaptation, environment or source changes.",
                  versions={k: importlib.metadata.version(k) for k in ["torch", "lerobot", "numpy"]},
                  star_saved_config=OmegaConf.to_container(cfg, resolve=True), constructors=constructor_configs(cfg),
                  source_files={key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key, path in FILES.items()},
                  temporal_checks=sentinel_and_windows(), existing_15000_outputs=real_outputs_and_normalization(stats),
                  actual_native_launch=actual_native_launch(),
                  checkpoint_buffers=checkpoint_buffers(Path(existing["checkpoint"])),
                  posterior_precision_existing_evidence={ep: info["posterior_losses"] for ep, info in existing["episodes"].items()},
                  comparison=[dict(item=row[0], starvla=row[1], native=row[2], interpretation=row[3], source_refs=row[4]) for row in comparison_rows()],
                  limitations=["No training executed by this audit; actual native launch settings checked from saved config/preprocessing/source, not future native model results.", "Prediction-vs-GT best temporal shift is descriptive, not evidence loader uses that shift.", "Existing NPZ cannot prove behavior of an external deployment client that was not supplied.", "Native re-training changes multiple variables; improvement alone cannot isolate cause or prove source actions wrong.", "Original data semantic/distribution audit belongs to Sagan; not repeated here."])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# StarVLA 15000步 → 本次原生LeRobot 0.3.3：只读配置/时间链路对照", "", "结论：未发现±16帧偏移或只重复pred[0]。保存的单chunk本身近乎常数。本次原生实际启动已核对：15k/batch8/FP32、ImageNet预训练+归一化、原生未加权L1、未改decoder、letterbox224。多变量同时变化，不能单独归因框架或数据。", "", "## 配置对照（本次实际启动；默认差异另作说明）", "", "| 项目 | 已完成StarVLA run | 本次原生配置/默认说明 | 解释/不可归因项 | 源码行号 |", "|---|---|---|---|---|"]
    lines.extend("| " + " | ".join(row) + " |" for row in comparison_rows())
    lines.extend(["", "## 可复核检验", "", "- 实际拼接表达式用37帧、16步、20维时间编码sentinel验证，跨0/16/32边界及末尾截断均正确；原生select_action队列输出逐项相等。", "- StarVLA wrapper真实方法执行sentinel解归一化，保留[B,16,20]所有时间步，没有repeat或只取step0。", "- ep0/45/90真实动作在0/52/104/末帧起点对照原生_get_query_indices与旧get_state_or_action，逐项一致；末帧仅padding mask不同。", "- 15000步npz中的GT/state与训练路径Parquet逐项一致；归一化/逆变换已在CPU核对。具体误差、shape、±16描述性MAE均见report.json。", "", "| Episode | FP32 chunk span中位数范围(rad) | GT span中位数范围(rad) | 实际concat与repeat0是否逐项相同 |", "|---|---:|---:|---|"])
    for ep, data in report["existing_15000_outputs"].items():
        m = data["modes"]["fp32"]
        lines.append(f"| {ep} | {min(m['median_chunk_span_right_arm']):.6g}–{max(m['median_chunk_span_right_arm']):.6g} | {min(m['median_gt_chunk_span_right_arm']):.6g}–{max(m['median_gt_chunk_span_right_arm']):.6g} | {m['concat_exactly_equals_repeat_step0']} |")
    lines.extend(["", "FP32单chunk有微小非零变化，BF16会进一步产生重复；台阶与repeat0相似不等于代码真的repeat0。", "", "## 同一内部源帧step0与GT平移的右臂MAE(rad)", "", "| Episode | GT t-16 | GT t | GT t+16 |", "|---|---:|---:|---:|"])
    for ep, data in report["existing_15000_outputs"].items():
        shifts = data["modes"]["fp32"]["same_rows_step0_mae_for_gt_offset"]
        lines.append(f"| {ep} | {shifts['-16']:.6f} | {shifts['0']:.6f} | {shifts['16']:.6f} |")
    lines.extend(["", "三段零平移均最准确，±16明显更差；结合索引/sentinel验证，不支持全局±16错位。不能排除未提供的部署客户端或少数帧小延迟。", "", "## 初步关键差异/独立变量", "", "优先关注：旧run强制BF16且KL低精度 vs本次FP32；旧random视觉初始化+FrozenBN identity/无ImageNet输入归一化 vs本次预训练+ImageNet归一化；旧整组action_model1e-4/右臂weighted loss vs原生两组1e-5/未加权L1。BN类型、decoder、16步绝对时间定义是共用项，不是独立框架差异。", "", "同15k/batch8/letterbox224仍存在LR/WD/betas/scheduler/clip、AMP、loss权重、padding、sampler、seed42/1000及原run恢复丢失Adam状态等混杂项。原生能否学到chunk动态是有效控制问题，但改善不能直接证明数据或action错误。", "", "本审计只读实际配置/source，未创建或修改训练配置/数据，未导入或调用launcher；make_dataset hook返回原生dataset，update/save先调原函数后追加日志/GT。40step smoke模型丢弃，官方train freshpolicy。没有依赖后续native loss/GT。", "", "## 源码绝对路径及SHA256", ""])
    lines.extend(f"- `{key}`: `{item['path']}` SHA256 `{item['sha256']}`" for key, item in report["source_files"].items())
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("NATIVE_ACT_CONFIG_AUDIT_OK " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
