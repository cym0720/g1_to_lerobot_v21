#!/usr/bin/env python3
"""Read-only, CPU-only audit of the current G1 ACT temporal learning path.

Uses source-extracted loader methods without dataset initialization/cache writes.
Only independent diagnostics under --output are written; no optimizer is run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
from PIL import Image

WSL_ROOT = Path("/home/cym/Projects_2026/starvla-g1-validation")
REPO = WSL_ROOT / "third_party/starvla"
RUN = Path("/home/cym/artifacts/starvla/runs/g1_act_0923_pick_apple_plate_16step")
CHECKPOINT = RUN / "checkpoints/steps_6000_pytorch_model.pt"
DATASET = Path("/home/cym/datasets/g1/0923_g1_pick_apple/empty_text_task_1")
WORKSPACE = Path(__file__).resolve().parents[2]
LOADER = REPO / "starVLA/dataloader/gr00t_lerobot/datasets.py"
FRAMEWORK = REPO / "starVLA/model/framework/VM4A/ACT.py"
STATE_ACTION = REPO / "starVLA/dataloader/gr00t_lerobot/transform/state_action.py"
HORIZON = 16
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def source_methods(path, class_name, names, namespace):
    """Execute exact selected methods, avoiding constructors/import side effects."""
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {n.name: namespace[n.name] for n in nodes}


def normalize(raw, stats, method):
    receiver = SimpleNamespace(mode="mean_std", statistics={k: torch.tensor(v) for k, v in stats.items()})
    return method(receiver, torch.as_tensor(raw, dtype=torch.float32)).numpy()


def audit_windows(stats, normalizer):
    namespace = dict(np=np, torch=torch, Image=Image)
    funcs = source_methods(LOADER, "LeRobotSingleDataset", ["get_state_or_action", "retrieve_data_and_pad", "_pack_sample"], namespace)
    meta = json.loads((DATASET / "meta/modality.json").read_text())
    action_meta = SimpleNamespace(**meta["action"]["robot"])
    receiver = SimpleNamespace(
        delta_indices={"action.robot": np.arange(HORIZON)},
        lerobot_modality_meta=SimpleNamespace(action={"robot": action_meta}),
        metadata=SimpleNamespace(modalities=SimpleNamespace(action={"robot": SimpleNamespace(absolute=True)})),
        get_trajectory_index=lambda _: 0,
    )
    receiver.retrieve_data_and_pad = MethodType(funcs["retrieve_data_and_pad"], receiver)
    # A time-encoded synthetic array detects accidental broadcasting/slicing.
    sentinel = np.arange(40, dtype=np.float32)[:, None] * 100 + np.arange(20, dtype=np.float32)
    receiver.curr_traj_data = pd.DataFrame({"action.qpos": list(sentinel)})
    receiver.trajectory_lengths = [len(sentinel)]
    synthetic = funcs["get_state_or_action"](receiver, 0, "action", "action.robot", 7)
    np.testing.assert_array_equal(synthetic, sentinel[7:23])

    df = pd.read_parquet(DATASET / "data/chunk-000/episode_000000.parquet")
    raw = np.stack(df["action.qpos"])
    receiver.curr_traj_data = df
    receiver.trajectory_lengths = [len(raw)]
    starts = [0, 52, 104, 156, len(raw) - 1]
    packer = SimpleNamespace(modality_keys={"video": ["video.test"], "action": ["action.robot"], "state": ["state.robot"], "language": ["language.test"]}, data_cfg={"include_state": True}, tag="new_embodiment")
    # Invoke the exact ACT batch builder with a tiny CPU parameter holder.
    from omegaconf import OmegaConf
    helpers = dict(np=np, torch=torch, Image=Image, OBS_STATE="observation.state", ACTION="action", List=list)
    tree = ast.parse(FRAMEWORK.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_image_feature_name", "_get_image_size", "_letterbox_image"}:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(FRAMEWORK), "exec"), helpers)
    helpers["ACTDefaultConfig"] = SimpleNamespace(image_keys=("test",), image_size=(8, 8))
    batch_builder = source_methods(FRAMEWORK, "ACT", ["_examples_to_act_batch"], helpers)["_examples_to_act_batch"]
    fake_model = SimpleNamespace(config=OmegaConf.create({"framework": {"image_keys": ["test"], "image_size": [8, 8], "chunk_size": HORIZON, "action_dim": 20, "state_dim": 20}}), action_model=torch.nn.Linear(20, 20))
    examples, rows = [], []
    for start in starts:
        window = funcs["get_state_or_action"](receiver, 0, "action", "action.robot", start)
        expected = raw[np.minimum(start + np.arange(HORIZON), len(raw) - 1)]
        np.testing.assert_array_equal(window, expected)
        normalized = normalize(window, stats["action"], normalizer)
        packed = funcs["_pack_sample"](packer, {"video.test": np.zeros((1, 8, 8, 3), dtype=np.uint8), "action.robot": normalized, "state.robot": np.zeros((1, 20)), "language.test": ["test"]})
        examples.append(packed)
        unquantized_span = np.ptp(normalized[:, 8:15], axis=0)
        quantized_span = np.ptp(packed["action"][:, 8:15].astype(np.float32), axis=0)
        rows.append(dict(start=start, source_indices=np.minimum(start + np.arange(HORIZON), len(raw) - 1).tolist(), raw_right_arm_span=np.ptp(window[:, 8:15], axis=0).tolist(), normalized_right_arm_span=unquantized_span.tolist(), after_float16_span=quantized_span.tolist(), max_float16_error=float(np.abs(normalized - packed["action"].astype(np.float32)).max())))
    batch = batch_builder(fake_model, examples)
    for i, sample in enumerate(examples):
        np.testing.assert_array_equal(batch["action"][i].numpy(), sample["action"].astype(np.float32))
    episodes = [json.loads(s) for s in (DATASET / "meta/episodes.jsonl").read_text().splitlines() if s.strip()]
    lengths = [ep["length"] for ep in episodes]
    padded_cells = sum(sum(max(0, step - length + 1) for step in range(HORIZON)) for length in lengths)
    # For short episodes, count directly; current episodes are all longer than H.
    padded_cells = sum(sum(max(0, start + HORIZON - length) for start in range(length)) for length in lengths)
    return dict(sentinel_exact_match=True, actual_episode_exact_match=True, act_batch_exact_match=True, final_shape=list(batch["action"].shape), actual_offsets=list(range(HORIZON)), includes_current_action=True, no_broadcast_or_wrong_slice=True, windows=rows, tail_padding=dict(episode_count=len(lengths), frame_count=sum(lengths), padded_target_cells=padded_cells, target_cell_fraction=padded_cells / (sum(lengths) * HORIZON), all_terminal_action_is_pad_false=bool(not batch["action_is_pad"].any()), note="Absolute actions repeat the final row. The ACT wrapper does not pass a real padding mask. This is a verified masking defect, not proof it causes interior collapse."))


def kl_formula(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()


def kl_test():
    cases = []
    for mu_value, logvar_value in [(0.01, 0.0), (0.0, 0.001), (0.0, -0.001), (0.003, 0.003), (0.01, 0.01), (0.0, 0.01), (0.0, -0.01), (0.0, 0.02), (0.01, 0.02)]:
        mu = torch.full((6, 32), mu_value).bfloat16().requires_grad_()
        logvar = torch.full((6, 32), logvar_value).bfloat16().requires_grad_()
        direct = kl_formula(mu, logvar)
        direct_grads = torch.autograd.grad(direct, (mu, logvar))
        fp32 = kl_formula(mu.float(), logvar.float())
        stable = 0.5 * (mu.float().square() + torch.expm1(logvar.float()) - logvar.float()).sum(-1).mean()
        safe_grads = torch.autograd.grad(stable, (mu, logvar))
        cases.append(dict(mu=mu_value, logvar=logvar_value, bf16=float(direct), fp32_same_rounded_input=float(fp32), fp32_expm1=float(stable), weighted_loss_error=float((direct.float() - stable) * 10), bf16_mu_gradient=float(direct_grads[0][0, 0]), safe_mu_gradient=float(safe_grads[0][0, 0]), bf16_logvar_gradient=float(direct_grads[1][0, 0]), safe_logvar_gradient=float(safe_grads[1][0, 0])))
    baseline = json.loads((WORKSPACE / "output/act_gt_audit_20260929/model_gt/report.json").read_text())
    observed = {}
    for ep, info in baseline["episodes"].items():
        observed[ep] = {mode: dict(l1=p["weighted_l1"], kl=p["kl"], fp32_same_mu_logvar_kl=p["kl_computed_fp32"], excess_loss_from_kl=10 * (p["kl"] - p["kl_computed_fp32"])) for mode, p in info["posterior_losses"].items()}
    return dict(synthetic_same_inputs=cases, observed_mainline_posterior=observed, note="Same rounded BF16 mu/logvar compared, so the discrepancy is arithmetic, not different model predictions. Gradient errors are also measured. FP32 expm1 avoids subtracting two near-one values.")


def decode_images(frame):
    import cv2
    images = []
    for camera in CAMERAS:
        cap = cv2.VideoCapture(str(DATASET / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"))
        try:
            assert cap.isOpened()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            assert ok
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            scale = min(224 / image.width, 224 / image.height)
            size = (round(image.width * scale), round(image.height * scale))
            resized = image.resize(size, Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (224, 224))
            canvas.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))
            array = np.asarray(canvas, dtype=np.float32) / 255
            images.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0))
        finally:
            cap.release()
    return images


def decoder_diagnostics(stats, normalizer, output):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACT
    features = {f"observation.images.{camera}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for camera in CAMERAS}
    features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(20,))
    config = ACTConfig(chunk_size=HORIZON, n_action_steps=HORIZON, input_features=features, output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))}, pretrained_backbone_weights=None)
    model = ACT(config).cpu().eval()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    prefix = "action_model.model."
    model.load_state_dict({k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}, strict=True)
    del checkpoint
    df = pd.read_parquet(DATASET / "data/chunk-000/episode_000000.parquet")
    state = np.stack(df["observations.state.qpos"])
    action = np.stack(df["action.qpos"])
    action_norm = normalize(action, stats["action"], normalizer).astype(np.float16).astype(np.float32)
    action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32)
    w = torch.full((20,), 0.1)
    w[8:16] = 8
    captured = {}
    def hook(_, args, kwargs):
        captured["x"] = args[0].detach()
        captured["memory"] = args[1].detach()
        captured["positions"] = kwargs["encoder_pos_embed"].detach()
    handle = model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
    observations = []
    saved_arrays = {}
    for frame in [52, 104, 156]:
        normalized_state = normalize(state[frame], stats["state"], normalizer).astype(np.float16).astype(np.float32)
        with torch.no_grad():
            model({"observation.images": decode_images(frame), "observation.state": torch.from_numpy(normalized_state).unsqueeze(0)})
        memory, positions = captured["memory"], captured["positions"]
        mean_memory = memory.mean(0, keepdim=True)
        spread_ratio = (memory - mean_memory).norm() / memory.norm()
        cosines = torch.nn.functional.cosine_similarity(memory[:, 0], mean_memory[0, 0].unsqueeze(0), dim=-1)
        target = torch.from_numpy(action_norm[frame:frame + HORIZON]).unsqueeze(0)
        variants = {}
        for variant in ["actual", "constant_memory", "query_as_content"]:
            model.zero_grad(set_to_none=True)
            query = model.decoder_pos_embed.weight.detach().clone().requires_grad_()
            x = torch.zeros_like(captured["x"])
            chosen_memory = memory
            if variant == "constant_memory":
                chosen_memory = mean_memory.expand_as(memory)
            elif variant == "query_as_content":
                x = x + query.unsqueeze(1)
            hidden = model.decoder(x, chosen_memory, decoder_pos_embed=query.unsqueeze(1), encoder_pos_embed=positions)
            prediction = model.action_head(hidden.transpose(0, 1))
            weighted_l1 = ((prediction - target).abs() * w).sum() / (target.shape[0] * target.shape[1] * w.sum())
            weighted_l1.backward()
            gradient = query.grad.detach()
            variants[variant] = dict(weighted_l1=float(weighted_l1.detach()), prediction_right_arm_span_raw=((prediction.max(1).values - prediction.min(1).values)[0, 8:15] * action_std[8:15]).detach().tolist(), query_gradient_norm=float(gradient.norm()), query_gradient_row_norms=gradient.norm(dim=1).tolist(), query_gradient_max_abs=float(gradient.abs().max()), query_grad_all_finite=bool(torch.isfinite(gradient).all()), self_attention_is_not_unused=bool(model.decoder.layers[0].self_attn.in_proj_weight.grad is not None))
            saved_arrays[f"frame_{frame}_{variant}_grad"] = gradient.numpy()
        observations.append(dict(frame=frame, encoder_tokens=int(memory.shape[0]), memory_relative_token_spread=float(spread_ratio), cosine_to_mean_min=float(cosines.min()), cosine_to_mean_median=float(cosines.median()), target_right_arm_span_raw=np.ptp(action[frame:frame + HORIZON, 8:15], axis=0).tolist(), variants=variants))
        print(f"CPU decoder frame {frame}: spread={float(spread_ratio):.8f}, gradient={variants['actual']['query_gradient_norm']:.8g}", flush=True)
    handle.remove()
    query = model.decoder_pos_embed.weight.detach()
    np.savez_compressed(output / "decoder_cpu_arrays.npz", query_embedding=query.numpy(), **saved_arrays)
    return dict(device="cpu", optimizer_steps=0, checkpoint=str(CHECKPOINT), decoder_layer_count=len(model.decoder.layers), query_embedding_shape=list(query.shape), query_embedding_std=float(query.std()), query_between_rows_rms=float((query - query.mean(0)).square().mean().sqrt()), query_parameter_requires_grad=bool(model.decoder_pos_embed.weight.requires_grad), initialization="nn.Embedding defaults; ACT _reset_parameters only reinitializes encoder/decoder matrices, not the query embedding", gradient_path="query -> attention q/k -> softmax weights -> weighted encoder values -> action head; zero decoder content eliminates a direct query residual path", frames=observations, note="Exact checkpoint decoder and real encoder memories, CPU float32, all modules eval, no optimizer. Query-as-content is a diagnostic intervention, not a validated repair or deployed architecture.")


def write_markdown(report, output):
    kl = report["kl"]["observed_mainline_posterior"]["0"]["autocast_bf16"]
    frames = report["decoder"]["frames"]
    lines = ["# G1 ACT training-side CPU audit", "", "## Confirmed findings", "", "- The exact loader and wrapper methods preserve action[t:t+16] as [B,16,20]. Offsets are 0..15 (includes current frame), not 1..16. No broadcast or final-window slicing was found.", "- The loader casts normalized state/action to float16, but the tested interior target windows still retain their temporal changes; this does not explain ~1e-4 rad predicted chunk spans.", "- train_starvla.py:420 forces CUDA BF16 autocast even when Accelerate is launched with mixed_precision=no. ACT.py:250 computes KL directly on BF16 posterior values.", f"- Mainline ep0 same posterior values: KL {kl['kl']:.9g} vs FP32 recompute {kl['fp32_same_mu_logvar_kl']:.9g}; excess composite loss {kl['excess_loss_from_kl']:.9g}. This is a confirmed arithmetic defect, not evidence of duplicated dataset actions.", "- Terminal actions repeat the last frame, while ACT.py:217-219 sets every action_is_pad=False. This is a real padding-mask defect; it cannot alone explain collapse in interior windows.", "", "## Decoder: evidence, not a blanket bug claim", "", "- LeRobot modeling_act.py:380 defines 16 distinct learnable query embeddings; :387-391 leaves their default nn.Embedding initialization intact. They require gradients and are covered by action_model's optimizer group (trainer_tools.py:127-139). No detach/freeze of these queries was found.", "- modeling_act.py:513-522 uses zero decoder content and query embeddings only as positional q/k inputs (:658-670). This is the shipped ACT/DETR-style design, not inherently a wiring bug.", "- With identical encoder values, changing attention weights cannot encode time; the constant-memory CPU intervention demonstrates loss of the query gradient. The real-memory results below measure how close this checkpoint is to that regime.", "", "| Frame | Encoder token spread / norm | Query grad norm | Constant-memory query grad | Query-as-content grad |", "|---|---:|---:|---:|---:|"]
    for row in frames:
        v = row["variants"]
        lines.append(f"| {row['frame']} | {row['memory_relative_token_spread']:.6g} | {v['actual']['query_gradient_norm']:.6g} | {v['constant_memory']['query_gradient_norm']:.6g} | {v['query_as_content']['query_gradient_norm']:.6g} |")
    lines.extend(["", "## Minimal controlled experiments", "", "1. Keep same checkpoint/data/sampling and BF16 network; compute KL in FP32 (prefer 0.5*(mu.float().square()+expm1(logvar.float())-logvar.float())), log weighted_l1 and kl separately. This isolates the loss arithmetic defect.", "2. Separately compare original zero decoder content vs query-as-content, identical FP32-safe KL and sampler. Measure held-out per-chunk span ratio and horizon-specific MAE, not only step-0 trajectory MAE. Mainline is already running this experiment.", "3. Only after those controls, test additional decoder depth or supervised temporal-delta loss. Do not replace action by future state merely to mask an architecture/precision problem.", "4. Propagate real terminal action_is_pad, or exclude incomplete windows as a small controlled check; keep this independent of the decoder experiment.", "", "## Source line references", "", "- Framework ACT.py:104-115,136-219,241-252", "- Loader datasets.py:1198-1204,1379-1416,1526-1571,1690-1745", "- Data registry G1ThreeCam/data_config.py:47-55 and G1_0923_PinkApple/data_config.py:3-6", "- LeRobot modeling_act.py:378-391,423-474,510-530,591-611,655-683", "- Trainer train_starvla.py:103-111,415-430; trainer_tools.py:92-148", "", "All source paths/versions, exact numerical outputs and scope limitations are in report.json. No original data, checkpoint, training script or CUDA process was modified."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output/act_gt_audit_20260929/training_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20260929)
    stats = json.loads((RUN / "dataset_statistics.json").read_text())["new_embodiment"]
    normalizer = source_methods(STATE_ACTION, "Normalizer", ["forward"], dict(torch=torch))["forward"]
    print("Checking actual loader/window/batch methods on CPU...", flush=True)
    report = dict(scope="CPU-only read-only diagnostics; output directory only; no optimizer/ training modifications", windows=audit_windows(stats, normalizer))
    print("Checking KL BF16 arithmetic and its gradients on CPU...", flush=True)
    report["kl"] = kl_test()
    print("Checking checkpoint decoder gradients using three real CPU encoder memories...", flush=True)
    report["decoder"] = decoder_diagnostics(stats, normalizer, args.output)
    import importlib.metadata
    report["versions"] = {key: importlib.metadata.version(key) for key in ["torch", "lerobot", "numpy"]}
    paths = [FRAMEWORK, LOADER, STATE_ACTION, REPO / "starVLA/training/train_starvla.py", REPO / "starVLA/training/trainer_utils/trainer_tools.py"]
    report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, args.output)
    print("TRAINING_CPU_AUDIT_OK " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
