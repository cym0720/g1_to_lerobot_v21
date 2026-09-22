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

旧 `state20/action22/opening` 只能作为输入兼容格式。新输出的夹爪是 `closed_fraction`：`0=open`，`1=closed`；旧 opening 输入必须显式转换为 `closed = 1 - opening`。

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

## 当前实现顺序

1. 已建立 EVA contract、通用 inclusive range-plan、夹爪候选段与 PCA/球证据融合函数。
2. 已实现通用 splitter：它先完整只读预检输入 Parquet 和视频帧数，再在新目录切分、重编号并重建 metadata/stats。
3. 下一步迁移 converter：按命名原始字段组装 qpos20，显式声明输入夹爪语义，写入新的 `dataset/converted/<id>`；缺失动作字段不得静默补零。
4. 再从真实乒乓球源代码/标注迁移球检测 adapter；不从编译缓存猜阈值。
5. 每个改写数据的阶段必须完成完整读取、帧数/metadata 一致性和 ACT loss gate。
