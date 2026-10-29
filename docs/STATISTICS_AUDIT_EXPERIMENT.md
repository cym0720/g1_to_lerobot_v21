# 统计链路审计与归一化对照实验

日期：2026-10-27

状态：实验设计；本文件不表示已经运行审计或训练。

## 问题与假设

目标是判断“模型效果差”是否由统计数据错误或归一化口径导致。需要分别检验：

1. **落盘统计计算错误**：`stats.json` / `episodes_stats.jsonl` 与训练所见 Parquet、视频不一致，或全局聚合权重/方差算法错误。
2. **统计来源泄漏/不一致**：训练归一化读到了 train 以外帧，或 val/test 使用了自己的统计，导致评估尺度和训练尺度不同。
3. **统计本身正确但覆盖不足**：train 的动作分位数不能覆盖 held-out 轨迹，归一化后被裁剪/饱和；统计重算正确也无法弥补数据覆盖缺失。
4. **数值统计正常、RGB 统计或预处理错配**：图像统计按错误颜色通道、尺度、letterbox阶段或帧集合计算，或 loader 又做了一次预处理。

不要用单一训练 loss 判断这些假设。先做与模型无关的重算审计，再对归一化做受控训练对照。

## 第一阶段：只读数据审计（不训练）

对象：`output/cleaned/head_color_fisheye_lr/1026` 及其训练实际使用的 train/val 视图和统计文件。以训练 manifest 中的 episode IDs 为准，不以目录 glob 作为训练集定义。

### A. 数值统计复算

- 按 episode manifest 读取每个 Parquet，直接从实际数值列逐帧复算 `count/min/max/mean/std/q01/q99`；均值、方差使用 float64，population std（`ddof=0`）。
- 全局统计从所有 train 帧拼接/流式合并得到，并与 `stats.json` 比较；再独立以 per-episode `count/mean/std` 合并，检查合并式的一致性。
- 分别审 `observation.state` 与 `action`，逐维对应命名的 20D contract。拒绝静默 reshape、错列、NaN/Inf、零方差、count 与总帧数不一致。
- 比较原始全量数据、train 子集、val/test 子集的分布。held-out 只用于覆盖率诊断，绝不参与候选训练统计。

### B. RGB 统计复算

- 选训练配置声明的全部相机，按 train manifest 解码其最终 letterbox 视频；复算 RGB `[0,1]` 的逐通道 min/max/mean/std 与像素数、视频帧数。
- 另外用训练 loader 实际输出的变换后 tensor 对同一固定帧抽样复算，记录是否 RGB/BGR、值域 `[0,1]` 或 `[-1,1]`、归一化前后、letterbox 是否重复。
- 黑色 padding 会显著拉低图像均值；因此既报告全画面统计，也报告有效内容区域统计。只在训练实际使用全画面标准化时，才以全画面统计作为候选。

### C. 训练消费路径追踪

- 对选定训练配置，记录最终加载的 stats 文件绝对路径与 SHA256、来源 split/episode、每个字段实际使用的 normalizer、epsilon、clip 范围及其应用先后。
- 用固定 batch 对 loader 的原始值、归一化值、反归一化值做 round-trip；有限值、维度/顺序匹配，数值误差容限 `1e-5`（低精度 tensor 单列说明）。
- 若 dataloader 在读取 manifest 前通过 parquet glob 自动算 stats，必须证明该 glob 可见范围仅含 train 轨迹，避免 held-out episode 混进归一化。

### 第一阶段判据

- **PASS**：计数和维度精确匹配；min/max 精确匹配（允许浮点存储误差 `1e-6`）；mean/std 相对误差不超过 `1e-5`；loader 消费文件 hash 和预期一致；round-trip 通过；无非有限值/零方差未处理维度。
- **FAIL**：任何样本来源、count、维度、语义顺序、统计公式、train-only 边界或 loader round-trip 不符。先修复数据/统计链路并重新做完整审计，再进行模型实验。
- **覆盖不足但统计正确**：数值复算 PASS，而 held-out 动作明显超出 train q01/q99 或实际 clip 边界。进入第二阶段验证影响。

## 第二阶段：受控归一化实验

使用现有 `1020+1026` fisheye 任务和已固定的 episode manifest；固定数据、图像、随机种子列表（至少 3 个）、模型结构、优化器、训练步数、batch、采样顺序策略、增强和 checkpoint 选择规则。测试集在方案冻结前保持封存；用 validation 选 checkpoint/参数，最后只评估一次 held-out test。

| 组别 | 归一化 | 目的 |
|---|---|---|
| A | 当前训练配置实际消费的统计 | 基线复现 |
| B | 从精确 train episode manifest 逐帧重算的统计；数值和 RGB 分开替换/启用 | 检验统计实现/来源差异 |
| C | 精确 train 统计，连续值按训练分位数缩放但关闭/放宽归一化裁剪（只用于动作输出维度） | 检验 q01/q99 clipping 饱和 |

若 A 与 B 的统计数组/hash 完全一致，合并 A/B，只跑 A 与 C。若 RGB loader 仅把图像缩放到 `[0,1]` 且模型没有按 RGB stats 标准化，RGB 统计不是当前训练因果变量，应从本轮训练对照排除，只做输入范围/相机映射审计。

### 指标

- 主指标：固定 test episodes 的 right-arm + right-gripper（qpos20 indices 8–15）反归一化 chunk MAE，按帧加权；也报告每 episode、每维 MAE/RMSE。
- 辅指标：validation 同指标、训练/验证 loss 曲线、预测动作的每维范围、归一化动作落在 clip 上下界的比例、GT 超出 train q01/q99 与 clip 边界的比例、预测饱和率。
- 统计报告提供每维 train/val/test min/max/q01/q99、有效样本数、episode 等权与帧加权结果；推断用 episode 配对 bootstrap 置信区间，不能把高度相关的帧当独立样本。
- 改善需在至少 3 个 seed 上方向一致，test MAE 相对下降至少 5%，且 episode 配对 95% CI 不跨 0；否则报告为证据不足。此阈值在查看 test 结果前冻结。

## 当前代码检查提示（待第一阶段验证）

- `src/clean_tools/05_recompute_metadata_stats.py` 会逐帧解码 RGB；默认主流程不传 `--compute-rgb-stats`，但默认实现本身仍走 PyAV 统计。
- `tools/conversion/05_recompute_metadata_stats.py` 明确保留转换器的 RGB 合约常数，只重算 Parquet 数值列。两者目的不同，不可只凭文件名认定谁正确。
- 两个实现都将全局数值 mean/std 按帧数合并；需要针对实际训练 episode manifest 和 loader 做独立复算，而不是假设实现正确。
- `tools/curation_deminF/materialize_dataset_views.py` 通过训练子集 Parquet 逐帧计算 mean/std/min/max/q01/q99，输出 subset-specific stats，并记录 hash；需要确认本轮 H20/其他训练代码是否真的读取该文件，还是自行 glob 重算。
- RGB 统计是否影响模型，取决于 loader 是否实际使用它们做图像标准化。不能仅因 `stats.json` 有 RGB 字段就推断它影响训练。

## 实验保护与交付

- 本实验只读源数据，不覆盖现有数据集、stats、checkpoint 或运行目录；所有报告和新 view 写到新的带日期目录。
- 不启动机器人执行。
- 未完成 train/val/test 数量与 episode manifest 核对、完整输入读校验、真实 loader normalizer 追踪前，不启动多组训练。
- 产物：数据审计 JSON/CSV、loader round-trip 报告、固定实验 manifest、每 seed 日志、逐 episode 指标、paired bootstrap 摘要和结论。结论明确区分“统计实现错误”“归一化范围问题”“没有统计因果证据”。
