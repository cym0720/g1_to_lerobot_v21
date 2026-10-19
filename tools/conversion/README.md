# 转换工具索引

## 通用入口

- `00_convert_618_to_lerobot.py`：原始录制到当前 EVA qpos20 契约。
- `03_sync_task_prompts.py`：同步任务提示词。
- `build_train_val_split.py`：完整轨迹级 train/val 划分；统计需随后重算。
- `merge_final_datasets.py`：成品数据拼接，只重编号，不重推导 action 或交换身体列；输入语义一致性需事先确认。
- `92_export_v21_to_v30.py`：显式版本导出。

## 特定批次的构建与兼容工具

以下脚本保留以追溯历史产物，不是主流程的必需步骤：

- `build_pick_apple_1008_split.py`、`letterbox_1008_split.py`：1008 构建与图像变换。
- `merge_pick_apple_1006.py`、`merge_pick_apple_1008_1015.py`：历史批次合并；后者包含 action 重推导和身体换位。
- `prepare_merged_split100_6.py`、`prepare_merged_starvla_1006.py`、`prepare_multigrasp_accepted.py`：特定实验的输入准备。
- `rename_state_action_columns.py`：旧模型输入列名兼容。
- `repair_g1_action_gripper_scale.py`：针对指定历史数据的夹爪尺度修复。

历史工具可能使用 `[lift, waist_pitch]`，与当前 EVA 契约 `[waist_pitch, lift]` 不同，
不能仅凭文件名或批次名决定顺序。显式原始顺序参数的改造见根目录 `TODO.md`。
本次整理保留现有转换语义，没有实现该待办，也没有修改实际数据。
