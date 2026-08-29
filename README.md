# 618data → LeRobot v2.1 清洗流水线

本仓库将原始 `dataset/618data` 转为按任务拆分的 LeRobot v2.1 数据集，并在转换结果上完成时长校验、全臂静止帧裁剪、起始 action 平滑、离群轨迹审查和最终统计重算。原始数据与最终输出均不纳入 Git。

## 目录

- `config/`：任务 ID 映射、相机和数据字段配置。
- `src/clean_tools/`：任务级清洗步骤与主流水线编排。`n- `packages/`：可复用的 LeRobot v2.1 路径工具、action 低通滤波、FFmpeg 编解码/探测基础设施。
- `tools/conversion/`：数据格式转换与可选 v2.1→v3.0 导出工具。
- `tools/testing/`：原始数据检查、LeRobot 全量读取与 ACT smoke test。
- `dataset/`：原始数据（Git 忽略）。
- `output/`：转换、清洗、报告和训练产物（Git 忽略）。

## 主流程

```powershell
# 从仓库根目录运行；默认使用 8 个 worker
powershell -ExecutionPolicy Bypass -File src/clean_tools/run_618_pipeline.ps1 `
  -InputRoot dataset/618data `
  -OutputRoot output/618_cleaning_trial `
  -Workers 8
```

主流程顺序如下：

1. `tools/conversion/00_convert_618_to_lerobot.py`：原始 618data 转为每任务一个 LeRobot v2.1 原始数据集。
2. `tools/testing/01_validate_lerobot_v21_read.py`：完整读取全部 Parquet 与视频，输出只读校验报告。
3. `src/clean_tools/02_trim_arm_static_frames.py`：用全部 14 个 arm joint 判断首尾静止帧，并保留两端各 5 帧空白帧。
4. `src/clean_tools/03_smooth_startup_actions.py`：对 action 采用因果低通：保留静止前 5 帧 + 起始运动 0.5 秒 + 后 5 帧交叉淡入，不改变 state、视频、时间戳。
5. `src/clean_tools/05_recompute_metadata_stats.py`：从最终数据重算 LeRobot v2.1 metadata 与统计信息。

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