# 乒乓球专项：两步检测

这个目录不直接改写 LeRobot 数据。它只产出分析证据和最终 `range-plan/v1`，实际 parquet/video 裁剪由通用 `tools/split_by_ranges.py` 完成。

```text
第一步：plan_by_gripper
  observations.state.qpos
    -> candidate_gripper_ranges.json

第二步：detect_pingpong_hit
  candidate_gripper_ranges.json + 相机视频中的橙球检测/跟踪 + PCA motion evidence
    -> range_plan.json

通用 splitter
  range_plan.json
    -> 裁剪后的 LeRobot 数据
```

候选区间不代表最终保留区间。第二步默认自动扫描高位和左腕相机：OpenCV 根据橙色、圆形外观找球候选，再用连续位置、大小和运动情况跟踪过滤；随后把检测帧与右臂关节 PCA 运动异常对齐。只有球检测帧和 PCA 异常在配置的帧距内对齐，且落在夹爪候选段中，才保留该段。

检测阈值和相机列表在 `config/detectors/pingpong_ball.json`；PCA 特征、异常阈值和球/PCA 最大帧距在 `config/detectors/pingpong_hit.json`。第二步会把逐相机检测帧、PCA 异常帧和对齐结果写入最终区间计划的 `evidence`，并检查视频帧数与 qpos 帧数一致，避免错位。

常规命令顺序：

```powershell
uv run python examples/pingpong/plan_by_gripper.py `
  --input <qpos20-lerobot-root> --output <candidate-plan.json>

uv run python examples/pingpong/detect_pingpong_hit.py `
  --input <qpos20-lerobot-root> --candidate-plan <candidate-plan.json> `
  --output <range-plan.json>

uv run python tools/split_by_ranges.py `
  --input <qpos20-lerobot-root> --plan <range-plan.json> --output <cropped-root>
```

若需复用外部检测器/人工审核结果，第二步仍可通过可选 `--ball-events` 接收 `ball-events/v1`。一般情况下不需要手工生成此文件，使用内置视频检测即可。该证据格式是：

```json
{
  "schema": "ball-events/v1",
  "contract": "eva-replay-g1-qpos20-v1",
  "episodes": [
    {"task": "task_name", "episode_index": 0, "frame_indices": [120, 121]}
  ]
}
```

内置检测器只负责从视频产生逐帧证据；不直接裁剪数据，也不单独决定最终区间。最终区间由第二步融合逻辑生成。

如果检测器或人工审核先产生 CSV，可用 `import_ball_events.py` 转换。CSV 固定三列：

```csv
task,episode_index,frame_index
task_name,0,120
```

它可选接收第一步候选计划以检查 task、episode 和帧边界；不会从 CSV 推断球是否真的存在。
