# 用途：使用 LeRobot v2.1 兼容运行时加载一个本地任务的数据，作为 ACT 训练前的最小读数验证。
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path('output/618_all_rgb_test/02_lerobot_cleaned/01_task_04')
dataset = LeRobotDataset('local/01_task_04', root=ROOT, episodes=[0], download_videos=False)
print(f'loaded_frames={len(dataset)} fps={dataset.fps}', flush=True)
print(f'camera_keys={dataset.camera_keys}', flush=True)
