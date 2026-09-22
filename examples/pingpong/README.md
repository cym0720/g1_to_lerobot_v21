# 乒乓球专项：两步检测

这个目录不直接改写 LeRobot 数据。它只产出分析证据和最终 `range-plan/v1`，实际 parquet/video 裁剪由通用 `tools/split_by_ranges.py` 完成。

```text
第一步：plan_by_gripper
  observations.state.qpos
    -> candidate_gripper_ranges.json

第二步：detect_pingpong_hit
  candidate_gripper_ranges.json + ball-events/v1 + PCA motion evidence
    -> range_plan.json

通用 splitter
  range_plan.json
    -> 裁剪后的 LeRobot 数据
```

候选区间不代表最终保留区间。第二步必须同时具有球事件和时间邻近的 PCA 击打事件，才保留对应夹爪候选段。

球检测器与第二步的稳定接口是：

```json
{
  "schema": "ball-events/v1",
  "contract": "eva-replay-g1-qpos20-v1",
  "episodes": [
    {"task": "task_name", "episode_index": 0, "frame_indices": [120, 121]}
  ]
}
```

球检测器只负责写这个证据 JSON；它不裁剪数据，也不决定最终区间。
