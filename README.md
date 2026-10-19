# 618data → LeRobot v2.1 清洗流水线

本仓库将原始 G1 记录转换为 EVA replay 的 LeRobot v2.1 qpos20 数据契约。原始输入放在 `dataset/raw/`，不可变转换基线放在 `dataset/converted/`；日常清洗链路只维护 `output/current/data` 一份工作数据。原始数据与最终输出均不纳入 Git。

## Windows 清洗环境

Windows 清洗代码使用仓库根目录的 `.venv`，由 `uv` 管理；不要再使用已废弃的
`.venv_lerobot_v21`。首次创建或依赖变更后运行：

```powershell
uv sync --extra dev
uv run pytest -q
```

随后通过 `uv run python <脚本路径>` 运行 Python 工具，或直接执行下方的 PowerShell
主流程。`ffmpeg` 是系统级依赖，需预先安装并能在 `PATH` 中执行 `ffmpeg -version`。
`requirements.txt`、`requirements-dev.txt` 仅供必须使用 pip 的场景；版本和锁文件以
`pyproject.toml`、`uv.lock` 为准。

## 目录

- `config/contracts/`：版本化的输出数据契约；新 G1 输出使用 EVA qpos20。
- `config/sources/`：按机器人定义原始字段映射；不按数据集日期分目录。
- `config/detectors/`：与机器人无关的检测器参数。
- `src/clean_tools/`：普通数据的可选清洗步骤与主流水线编排。
- `packages/`：可复用的契约、范围计划、检测、LeRobot 路径和 FFmpeg 基础设施。
- `tools/`：薄 CLI；`tools/split_by_ranges.py` 只消费通用区间计划。
- `examples/pingpong/`：乒乓球专项的两步检测示例。
- `tools/testing/`：原始数据检查、LeRobot 全量读取与 ACT smoke test。
- `dataset/`：原始数据（Git 忽略）。
- `output/`：转换、清洗、报告和训练产物（Git 忽略）。
- `experiments/h20/`：远端训练与部署快照，不属于本机清洗主流程。
- `tools/diagnostics/h20/`、`tools/transfer/h20/`：历史评估和下载脚本。
- `artifacts/`：下载包、二进制与评估产物（Git 忽略）。

通用与特定批次转换工具的区别见 [转换工具索引](tools/conversion/README.md)。
根目录原 `scripts_stage/`、训练入口、`fetch_*.sh` 和临时评估脚本已归入上述目录；
旧部署打包器保存在 `tools/legacy/deployment/`。这些历史脚本需要其原远端环境。

StarVLA 训练与验证已移至独立 WSL 工作区
`/home/cym/Projects_2026/starvla-g1-validation`。该工作区使用 `uv`、独立
Git 历史和 WSL 的 CUDA 环境；初始化模板和迁移脚本分别位于
`docs/starvla/wsl_workspace_template/`、`tools/bootstrap_wsl_starvla_workspace.sh`。
本仓库只保留数据清洗代码，不再把 StarVLA 运行环境或历史部署包放在根目录。

详细的环境、普通/乒乓球运行顺序和完成门禁见 [运行手册](docs/PIPELINE_OPERATIONS.md)；历史
输出的归档与删除规则见 [输出保留规则](docs/OUTPUT_RETENTION.md)；本机原始数据的位置见
[数据目录约定](docs/DATA_LAYOUT.md)。

## 主流程

```powershell
# 从仓库根目录运行；默认使用 8 个 worker
powershell -ExecutionPolicy Bypass -File src/clean_tools/run_618_pipeline.ps1 `
  -InputRoot dataset/618data `
  -ConvertedRoot dataset/converted/618_qpos20 `
  -OutputRoot output/current `
  -Workers 8
```

上例采用严格 action 契约。仅当已审计并接受“缺失的 head/waist action 使用同帧 state 回退”时，才额外传入
`-AllowStateActionFallback`；该选择会记录到输出 provenance。当前 618 原始数据按计划保留，不执行这条批量转换命令。

主流程顺序如下：

1. `tools/conversion/00_convert_618_to_lerobot.py`：写入 `observations.state.qpos` 和 `action.qpos`，各为 float32[20]；默认读取 `config/sources/g1.json` 中已审计的录制器夹爪语义，只有切换录制器时才通过 CLI 显式覆盖。
2. 可选 `02_trim_arm_static_frames.py`：使用 qpos 中 14 个手臂关节裁剪首尾静止帧。
3. 可选 `03_smooth_startup_actions.py`：只平滑 `action.qpos`；默认关闭。
4. `05_recompute_metadata_stats.py`、完整读取验证和 ACT loss smoke：是每次物化数据后的必经门禁。

乒乓球数据只在 common tail 之前增加两步：先由 gripper 产生候选区间，再由球检测加 PCA 击打证据产生最终 `range-plan/v1`。随后 `tools/split_by_ranges.py` 完成一次通用裁剪；统计和校验不属于乒乓球的第三个检测步骤。

`src/clean_tools/04_review_outliers.py` 用于单独生成离群轨迹审查视频；确认删除清单后才会执行选择性删除。

## 数据集输出

每一个任务目录是独立的 LeRobot v2.1 数据集，包含：

```text
data/chunk-000/episode_*.parquet
videos/chunk-000/<camera_field>/episode_*.mp4
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/stats.json
```

当前转换所有非深度相机流：头部 RGB、左右腕部 RGB、头部中心/左/右鱼眼 RGB；深度流明确不转换。
## TODO

腰部原始顺序显式声明等待办见 [TODO.md](TODO.md)。目前顺序检测仍为数值启发式，
无法判定时会保留原始顺序；转换成功本身不能证明关节语义正确。

- [ ] 低通滤波：基于代表性遥操轨迹确定 action 起始段平滑的截止频率、持续时间和交叉淡入参数，并完成前后动作曲线与 ACT 训练效果对比。
- [x] FFmpeg 解码加速（已接入当前分支）：H.265 通过 `packages/ffmpeg_io.py` 进行时间戳对齐、FFmpeg 解码/转码，支持 CUDA → QSV → CPU 回退和输出帧数校验；转换器按所有输入任务统一调度受限进程池，设置解码线程数。后端吞吐测试入口为 `tools/testing/benchmark_raw_video_backends.py`，Windows 进程路由测试已通过。具体硬件上的加速比与完整读取结果以对应运行报告为准。
- [x] RGB 通道统计（已实现）：`packages/ffmpeg_io.py::rgb_video_stats()` 通过 FFmpeg `rgb24` 分块流式计算归一化 R/G/B 的 `mean`、`std`、`min`、`max`，内部按像素数累加、不缓存整段视频。运行 `src/clean_tools/05_recompute_metadata_stats.py --input <数据集目录> --compute-rgb-stats` 使用 FFmpeg 路径，结果写入 `meta/episodes_stats.jsonl` 和 `meta/stats.json`；不传该参数则使用原 PyAV 统计路径。对比测试入口为 `tools/testing/benchmark_rgb_stats.py`。
