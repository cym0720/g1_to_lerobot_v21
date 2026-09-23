# 618data → LeRobot v2.1 清洗流水线

本仓库将原始 G1 记录转换为 EVA replay 的 LeRobot v2.1 qpos20 数据契约。原始输入放在 `dataset/raw/`，不可变转换基线放在 `dataset/converted/`；日常清洗链路只维护 `output/current/data` 一份工作数据。原始数据与最终输出均不纳入 Git。

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

## 主流程

```powershell
# 从仓库根目录运行；默认使用 8 个 worker
powershell -ExecutionPolicy Bypass -File src/clean_tools/run_618_pipeline.ps1 `
  -InputRoot dataset/618data `
  -ConvertedRoot dataset/converted/618_qpos20 `
  -OutputRoot output/current `
  -AllowStateActionFallback `
  -Workers 8
```

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

- [ ] 低通滤波：基于代表性遥操轨迹确定 action 起始段平滑的截止频率、持续时间和交叉淡入参数，并完成前后动作曲线与 ACT 训练效果对比。
- [ ] FFmpeg 解码加速：在 `experiment/process-parallel-conversion` 分支中，以 FFmpeg 替代 Python/PyAV 的 H.265 逐帧解码；实现跨任务全局进程池、限制单进程线程数，并以相同输入测试吞吐、输出帧数与 LeRobot v2.1 完整读取一致性。
- [ ] RGB 通道统计：在 FFmpeg 解码/转码链路中以 `rgb24` 原始帧流传给流式累加器，计算每路相机的 R/G/B `mean`、`std`、`min`、`max` 和像素计数；不缓存整段视频，不影响 LeRobot 视频输出。默认关闭，仅在 `--benchmark` 或显式 `--compute-rgb-stats` 时运行；结果写入 benchmark 报告，验证后再用于最终 `stats.json`。
