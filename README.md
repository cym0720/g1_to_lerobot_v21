# AgiBot World → LeRobot 清洗流水线

本项目针对 AgiBot World Alpha 的 `observations/` + `meta_info/` 样例目录。数据先按**任务目录**转换成独立的 LeRobot v2.1 数据集；随后步骤 1、2、3 也始终按任务独立处理。

官方 AgiBot World 仓库推荐 [any4lerobot](https://github.com/Tavish9/any4lerobot) 做 AgiBot→LeRobot 转换。本项目已将其下载到 `vendor/any4lerobot` 作为字段和格式参考。由于当前样例使用 Alpha 的 `meta_info/<task>/<uuid>/proprio_joints.h5` 布局，而该工具当前版本面向另一版 `proprio_stats` 布局，`0_convert...py` 是兼容这一样例的适配器。

> 所有脚本默认输出新目录，绝不修改原始数据。

## 顺序

```powershell
python clean_tools/01_convert_agibot_alpha_to_lerobot.py --src sample_dataset/sample_dataset --output output/00_lerobot
python clean_tools/02_filter_short_episodes.py --input output/00_lerobot --output output/01_length_filtered --min-frames 100
python clean_tools/03_trim_arm_static_frames.py --input output/01_length_filtered --output output/02_trimmed --arm-joint-indices 15,16,17,18,19,20,21,22,23,24,25,26,27,28
python clean_tools/04_review_outliers.py --input output/02_trimmed --review-dir output/03_review --tail-frames 30
# 编辑 output/03_review/<task>/delete_list.txt 后：
python clean_tools/04_review_outliers.py --input output/02_trimmed --review-dir output/03_review --apply
```

`--arm-joint-indices` 必须对应实际 34 维关节向量中左右机械臂的索引。示例为常见 G1 排列的暂定值；请在投入完整数据前依据设备关节顺序确认。静止定义严格为：该列表中的**每一个**关节相邻帧变化均不超过阈值。

## 输出格式

每个任务成为一个独立 LeRobot v2.1 根目录，含 `data/chunk-000/*.parquet`、`videos/chunk-000/.../*.mp4`、`meta/info.json`、`meta/episodes.jsonl`、`meta/stats.json`。原始 H5 中全部非空 state/action 时序字段会写进 Parquet；RGB 视频原样复制。深度 PNG 会作为 `depth_raw/` 附属模态保留（LeRobot v2.1 对深度序列没有统一的视频编码约定），步骤 1、2、3 都会同步处理它。
