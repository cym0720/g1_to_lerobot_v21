# EVA G1 数据重构计划

## 目标输出

所有新 G1 LeRobot 数据使用 `config/contracts/eva_replay_g1_qpos20_v1.json`：

```text
observations.state.qpos  float32[20]
action.qpos              float32[20]
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
timestamp                float32[1]
```

旧 `state20/action22/opening` 只能作为输入兼容格式。新输出的夹爪是 `closed_fraction`：`0=open`，`1=closed`；旧 opening 输入必须显式转换为 `closed = 1 - opening`。当前 618 G1 原始编码器已确认 `0=open`、`120=closed`，因此 source 配置采用 `raw_encoder_0_120`，转换公式为 `clip(raw, 0, 120) / 120`。

## 两条管线

普通数据：

```text
raw -> G1 adapter -> EVA qpos20 converted baseline -> output/current/data
    -> optional trim/action smoothing -> stats -> read/video/ACT validation
```

乒乓球数据：

```text
EVA qpos20 converted baseline
  -> 第一步夹爪候选段
  -> 第二步球检测 + PCA 击打证据
  -> 最终 range plan
  -> 通用 splitter 一次性裁剪
  -> stats -> read/video/ACT validation
```

乒乓球专项只有上述两个检测步骤。统计、metadata 重建和校验是通用裁剪收尾，不是第三个检测步骤。

## 已完成的基础设施

1. 已建立 EVA contract、G1 原始字段映射、inclusive `range-plan/v1`、夹爪候选段与 PCA/球证据融合函数。
2. `tools/conversion/00_convert_618_to_lerobot.py` 已改写为 qpos20 输出：默认从 `config/sources/g1.json` 读取已审计的 618 夹爪规则（`0=open`、`120=closed`）；只有切换到另一种 G1 录制器时才通过 CLI 覆盖它。缺失 head/body action 默认拒绝，只有 `--allow-state-action-fallback` 才能回退到 state，并写入 provenance。
3. `tools/split_by_ranges.py` 已实现通用切分：先完整只读预检源 Parquet 与视频，再在新目录重写片段、索引、metadata 和 stats。
4. 普通数据入口 `run_618_pipeline.ps1` 已采用 `dataset/converted/<id>` 基线和 `output/current/data` 唯一工作副本。trim 与 action smoothing 都是显式 opt-in。
5. qpos20 的 read/video gate、统计重算和 ACT 真实优化 smoke 已迁移；ACT 适配层只在内存把 qpos 映射到当前 LeRobot 版本硬编码的 batch key，不向磁盘写回旧字段。

## 配置与参数政策

- 数据契约、机器人字段映射、检测阈值属于 `config/`，不放入业务脚本的 CLI。
- 模块内稳定的算法默认值应定义为命名常量，并在 docstring 或注释中说明单位和影响。
- CLI 只保留输入/输出、并发度、`--dry-run`、`--debug` 与会改变数据含义的显式选择。录制器已知的夹爪编码由 source 配置给出，且必须在 provenance 中落盘；对不同录制器的 CLI 覆盖仍是显式语义选择。
- `--allow-state-action-fallback` 绝不作为默认值：它会把缺失的 action head/waist 替换为同帧 state，适合已审计的旧录制器兼容，不能被误称为原始 action。

## 当前 618 批量预检结论（只读，2026-09-23）

对 `dataset/618data` 的 110 个 episode 执行转换器 `--dry-run`：

- 严格 action 契约：接受 0，拒绝 110；主要原因是每条都缺 `action/head/position` 和 `action/waist/position`。
- 加 `--allow-state-action-fallback`：接受 86，拒绝 24。剩余拒绝来自缺少 `record/raw_joints.h5`、`meta_info.json` 或 `action/joint/position`，这些不能回退，仍必须排除。

因此批量物化之前必须由数据负责人明确接受“head/waist action 回退”这一语义，并保留每条轨迹的 `conversion_validation.csv` 与 `meta/info.json` provenance；没有该决定时，主入口保持严格拒绝。

## 剩余迁移 / 淘汰清单

1. 从真实乒乓球球检测代码或人工标注迁移 `ball-events/v1` adapter；不从编译缓存猜阈值。它只负责第二步的球帧事件，不裁剪数据。
2. 已将旧字段依赖的 overlay、三相机抽取、旧 ACT loader 与 action-filter benchmark 移到 `tools/legacy/`，并记录正式替代关系。新增 legacy 工具不得被主入口引用。
3. 将转换器里仍然存在的早期兼容参数和重复媒体编码细节进一步下沉到 packages；每次移动后保留 Windows process-pool 测试。
4. 已用单条真实 618 episode 写入 `dataset/converted/g1_qpos20_encoder_smoke`：466 帧、三路视频，完整 read/video 校验通过，ACT 12-step loss gate 通过。下一步批量物化前仍应保留 `--allow-state-action-fallback` 的 provenance 审计。
