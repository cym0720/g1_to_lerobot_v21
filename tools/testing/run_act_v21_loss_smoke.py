# 用途：用 LeRobot v2.1 兼容运行时进入官方 ACT 训练流程，记录真实训练 loss，并验证 loss 的末段均值低于首段均值。
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


LOSS = re.compile(r"\bloss\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official LeRobot v2.1 ACT loss-decrease smoke test.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="单个 LeRobot v2.1 task 目录")
    parser.add_argument("--repo-id", default="local/act_v21_smoke")
    parser.add_argument("--output", type=Path, required=True, help="训练产物目录，必须不存在")
    parser.add_argument("--steps", type=int, default=30, help="真实 optimizer 更新步数，至少 10")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="安装 LeRobot 0.3.x 的 Python")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"), help="ACT smoke device")
    args = parser.parse_args()
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device
    if args.steps < 10:
        raise SystemExit("--steps 必须至少为 10，才可以判断 loss 趋势")
    if not (args.dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"不是 LeRobot task 根目录: {args.dataset_root}")
    if args.output.exists():
        raise SystemExit(f"训练输出已存在，拒绝覆盖: {args.output}")

    command = [
        str(args.python), "-u", "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.dataset_root}",
        "--policy.type=act",
        f"--policy.device={device}",
        "--policy.pretrained_backbone_weights=null",
        "--policy.push_to_hub=false",
        f"--output_dir={args.output}",
        f"--steps={args.steps}",
        "--batch_size=1",
        "--num_workers=0",
        "--eval_freq=0",
        "--log_freq=1",
        "--save_checkpoint=false",
        "--wandb.enable=false",
    ]
    environment = os.environ | {"HF_HUB_OFFLINE": "1", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.parent / f"{args.output.name}.log"
    losses: list[float] = []
    with log_path.open("w", encoding="utf8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace", env=environment)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            for value in LOSS.findall(line):
                losses.append(float(value))
        returncode = process.wait()
    if returncode:
        raise SystemExit(f"LeRobot ACT 训练失败，exit_code={returncode}，日志: {log_path}")
    if len(losses) < 10:
        raise SystemExit(f"训练完成但只提取到 {len(losses)} 个 loss，无法判断趋势；日志: {log_path}")
    width = max(3, len(losses) // 4)
    first = sum(losses[:width]) / width
    last = sum(losses[-width:]) / width
    passed = last < first
    summary = args.output.parent / f"{args.output.name}_loss_check.txt"
    summary.write_text(
        f"steps={args.steps}\nlosses={len(losses)}\nfirst_window_mean={first:.8g}\nlast_window_mean={last:.8g}\nloss_decreased={passed}\nlog={log_path}\n",
        encoding="utf8",
    )
    print(summary.read_text(encoding="utf8"), end="")
    if not passed:
        raise SystemExit("ACT 已执行，但 loss 末段均值没有下降；测试不通过")


if __name__ == "__main__":
    main()
