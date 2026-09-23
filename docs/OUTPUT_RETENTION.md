# 输出保留与归档规则

`output/` 是可再生的工作目录，只允许保存当前一次流水线运行：`output/current/`。它不用于
保存历史数据集、训练权重或调试产物，也不纳入 Git。

## 目录职责

| 位置 | 内容 | 保留规则 |
| --- | --- | --- |
| `dataset/raw/` | 原始采集数据 | 不修改、不删除 |
| `dataset/converted/` | 不可变转换基线 | 按数据契约保留 |
| `output/current/` | 当前流水线的临时工作结果 | 下一次运行前清理或归档 |
| `D:\go work\archives\data-cleaning\` | Windows 历史 LeRobot 输出与唯一压缩包 | 只读归档，不参与流水线 |
| `~/artifacts/starvla/runs/` | WSL 的 StarVLA 权重和训练日志 | 只在 WSL 侧使用 |

## 清理判定

- 有 `meta/info.json` 的 LeRobot 输出：先移入归档，不能直接删除。
- 唯一 ZIP：移入归档；只有已验证和同名目录重复的 ZIP 可以删除。
- `*.log`、`*.stderr.log`、`*.stdout.log`、dry-run 计划、FFmpeg smoke 视频：属于可再生调试证据，
  完成归档记录后删除。
- StarVLA checkpoint：先迁移到 WSL 的 artifacts 并核对逐文件内容清单，之后才删除 Windows 副本。

## 2026-09-23 历史整理

本次将 `output/` 中的 2026-08 至 2026-09 历史 LeRobot 目录迁至
`D:\go work\archives\data-cleaning\output-legacy-2026-09-23\lerobot\`；StarVLA 训练结果迁至
`~/artifacts/starvla/runs/legacy-windows-2026-09-23/`。根目录 `output/` 清空后重新作为
`output/current/` 的工作位置。
