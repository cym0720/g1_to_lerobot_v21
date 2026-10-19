# G1 数据清洗运行手册

本仓库只负责 Windows 侧的数据转换、质量检查和乒乓球切分；StarVLA 训练与验证在
WSL 工作区 `/home/cym/Projects_2026/starvla-g1-validation` 执行。

## 先准备环境

在仓库根目录执行：

```powershell
uv sync --extra dev
uv run pytest -q
ffmpeg -version
```

`uv` 创建的 `.venv` 是唯一的 Windows Python 环境。请用 `uv run python ...` 运行单个
Python 工具；`run_618_pipeline.ps1` 会自动定位该环境。FFmpeg 是系统依赖，不能由
Python requirements 替代。

## 普通数据链路

普通数据不需要乒乓球检测。转换输入位于 `dataset/raw/`（具体批次布局见
[本机数据目录约定](DATA_LAYOUT.md)）。其顺序固定为：原始 HDF5 预检 -> EVA qpos20 转换 -> 可选
静止帧裁剪 -> 可选起始 action 平滑 -> 统计重算 -> LeRobot 完整读取验证 -> ACT loss smoke。

```powershell
powershell -ExecutionPolicy Bypass -File src/clean_tools/run_618_pipeline.ps1 `
  -InputRoot dataset/raw/<source> `
  -ConvertedRoot dataset/converted/<run_name> `
  -OutputRoot output/current `
  -Workers 8
```

默认不执行 action 平滑和静止帧裁剪；它们是需在报告中记录的质量选择，而非默认修改。
原始数据不可覆盖，`ConvertedRoot` 和 `output/current/data` 已存在时脚本会停止。

## 乒乓球链路

乒乓球仅比普通链路多两步，且两步都在通用统计与验证之前：

1. 夹爪候选：从 qpos 中找出可能的击打区间。
2. 球检测 + PCA：第二步自动用 OpenCV 扫描高位和左腕相机中的橙色乒乓球并做连续跟踪，再与右臂关节 PCA 运动证据对齐，输出 `range-plan/v1`。外部检测器或人工审核事件可选通过 `--ball-events` 输入，不是默认前置步骤。

球检测帧、PCA 异常帧及其对齐证据都会写进最终区间计划。随后由 `tools/split_by_ranges.py` 消费区间计划并物化裁剪后的独立 LeRobot episode；之后
复用普通链路的统计、完整读取和 ACT smoke 门禁。PCA 和通用切分器在 `packages/`，而乒乓球
检测及专项组合逻辑在 `examples/pingpong/`。

## 数据契约与夹爪语义

输出契约是 `config/contracts/eva_replay_g1_qpos20_v1.json`：state/action 都是 float32[20]。
原始 G1 的夹爪编码由 `config/sources/g1.json` 声明：`0` 为开、`120` 为闭；转换时会明确写入
provenance。缺失 head/waist action 只能显式传入 `-AllowStateActionFallback` 后回退同帧 state。

## 完成门禁

一次实际数据物化至少应保留下列证据：预检 CSV、转换报告、重算后的 `meta/stats.json`、完整
读取报告，以及 ACT loss smoke 输出。进程仍在运行、只有日志或只生成目录都不代表任务完成。
