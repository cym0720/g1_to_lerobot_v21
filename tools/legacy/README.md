# Legacy tools

这里保留的是历史实验或旧 LeRobot 字段契约工具，**不得**用于 EVA G1 qpos20 正式数据。

- `conversion/01_overlay_g1_rollflow20.py`：旧 `observation.state` / `action` 覆盖式转换，会改写现有数据。
- `conversion/02_extract_g1_three_camera.py`：旧三相机子集/状态拼接流程；qpos20 转换器已直接输出三相机。
- `testing/run_act_v21_native_smoke.py`、`run_act_v21_loss_smoke.py`、`act_v21_smoke_loader.py`：针对旧字段或旧 v2.1 loader 的 ACT 实验。
- `testing/benchmark_startup_filter.py`：针对旧 action 列的滤波实验。

正式替代关系：

```text
转换       tools/conversion/00_convert_618_to_lerobot.py
完整读取   tools/testing/01_validate_lerobot_v21_read.py
ACT gate   tools/testing/act_direct_loss_smoke.py
区间裁剪   tools/split_by_ranges.py
```

若要复用 legacy 思路，必须先迁移到 `observations.state.qpos` / `action.qpos`，并通过完整 read/video/ACT gate；不能仅修改 metadata 假装兼容。
