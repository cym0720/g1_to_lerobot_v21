# 时间对齐与模型抖动排查（2026-10-20）

> 历史实验记录：文中 `test/` 下的一次性诊断脚本已于 2026-10-29 清理；本文件保留当时的方法摘要和结果，不代表这些脚本仍可直接运行。

## 已完成的检查及边界

只读检查本地 `output/merged/0923+1004+1015+1019_pick_apple/dp_letterbox224`，
共 train 137 / 56,726 帧，val 15 / 3,311 帧。152 条均已回查原始 H5 与头相机时间戳，
按 episodes.jsonl 的 trim_source_start/end 还原原始帧范围。仅统计右臂，夹爪语义沿用已确认规则。
最初这一步没有解码全量视频或训练；后续视频完整性、推理和隔离短续训见文末各节。
正式数据、checkpoint与H20环境未修改；实验仅新增独立诊断输出。短续训不是完整重训或正式发布。

工具：`test/audit_time_alignment.py`。报告目录：
`output/diagnostics/time_alignment_20261020_203853_185176/`。
`summary.json`、`episodes.jsonl`、`gaps.csv` 保留可复查指标。
这里的 OK 表示诊断运行成功，不表示物理同步或模型效果已验收。

## 一起阅读的转换代码

文件：`tools/conversion/00_convert_618_to_lerobot.py`

1. 383 行：以 `camera/head_color/head_color.txt` 为 target。原始时钟由
   `packages/raw_preflight.py:26` 读取文本首列整数纳秒，不是视频 CFR 时间。
2. 56 行 `nearest()`：搜索插入点左右两条记录，选择绝对时间差较小的一条；相等时选择左侧。
   两端 clip，因此区间外也有返回值。没有检查最近距离上限，也没有区分 action/state。
3. 66 行 `h5_series_required()`：读取数据/时间戳，按二者最短长度截断，再对 target 最近邻取值。
   没有对内部秒级空档专门拒绝、报警或插值；检查有限值和向量宽度不能发现时序语义问题。
4. 131 行：可选 `joint_boundary_policy == leading_state_trailing_hold` 分支会用同帧 state
   回填头部边界之前的手臂 action；是否生效取决于实际 source 配置。不能认为所有历史输出都有此分支。
5. 217 行 `encode_wrist()`：腕部 JPG 文件名作为时间戳，每个 target 最近邻选图，可能复用图片。
   H265 则在 179 行调用 `packages/ffmpeg_io.py:196` 的对齐编码函数。
6. 393、440 行：写入时间戳改为 `arange(N)/fps`，图片也编码成固定帧率。帧内 action/image
   仍按原始头时钟配对，但不规则真实间隔被压成统一时间步。区分“帧内匹配”与“跨帧时间尺度”。

核心逻辑等价于：

```python
target = head_camera_timestamps_ns
right = searchsorted(action_timestamps_ns, target)
left = clip(right - 1)
right = clip(right)
chosen = where(abs(action_ts[right] - target) < abs(action_ts[left] - target), right, left)
action_for_frame = raw_action[chosen]
output_timestamp = arange(len(target)) / 30
```

若前后命令间隔 3 秒，空档后半段会使用下一条命令；这不是插值。
最近已发命令版本则是 `searchsorted(action_ts, target, side="right") - 1`；
在第一条命令之前必须单独制定边界政策，不能悄悄 clip 成 0 并宣称满足因果约束。

## 全量结果

|批次|轨迹|含保留区间 action 空档的轨迹|空档数|最近邻 vs 最近已发命令右臂平均差 rad|最大差 rad|
|---|---:|---:|---:|---:|---:|
|0923|91|56|60|0.000854|0.02534|
|1004|15|0|0|0.000798|0.01920|
|1015|40|18|19|0.000607|0.01747|
|1019|6|6|33|0.000651|0.02075|

空档阈值 100ms 只是诊断标记，不等于漏包。最近邻/previous 平均差仅在 action 原始覆盖区间计算，
差值不能判定哪个策略更符合执行语义。存储 action 与原始最近邻最大差 1.2e-7 rad，符合 float32 舍入。
最终保留帧中有 3,299 帧落在机械臂 action 内部空档。

训练集有 195 帧位于首条 action 之前、1,224 帧位于末条 action 之后；验证集分别 13、138 帧。
这些帧也有标签，主要由端点复用产生，不能当作有同期原始 action 记录。

明确边界案例：val episode 6，UUID `7dcfc220-f6b7-49de-9789-b2f921b7157d`，58 帧全在
最后一条机械臂 action 之后；该 action 在头相机开始前约 79ms 已结束。是否正常保持仍需采集语义证明。

本次 152 条原始头 RGB 时钟没有 >100ms 的间隔，但不等于没有任何丢帧或曝光延迟。
0923 中最差轨迹相对 30Hz 的累计时间偏差约 99.5ms；相隔 16 帧的持续时间误差，其单轨迹
p99 最大约 85.6ms。它会改变 action chunk 的实际时间跨度，但没有证明足以造成当前抖动。

多数轨迹 action[t] 最接近 state[t+3]：0923 中 77/91、1015 40/40、1019 6/6。
这是运动段 -10..10 帧同一目标集合的位置 MAE 扫描，不是独立的相机同步标定。
执行器响应/平滑本来会产生领先关系，不能据此把标签整体平移 3 帧。

## H20 的训练采样路径

经 H20 当前环境 CPU 导入注册表确认：

`g1_pick_apple_0923plus1004plus1015plus1019_train_c16`
→ `train39` → `g1_pick_apple_act20`
→ `_pickapple_multigrasp_base_data_config.PickAppleG1ACTConfig`。

类定义源文件：
`/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/baselines/starvla_local/examples/realRobots/PickAppleACT/train_files/data_registry/data_config.py`。

生效配置为 video/state `[0]`、action `[0,1,...,15]`、CHUNK=16。
基类 `gr00t_lerobot/datasets.py::get_state_or_action()` 使用 `base_index + delta_indices`，
不跨 episode，绝对动作越界复制首/末值。缓存视频使用相同 base_index 加对应轨迹 offset。
但视频缓存的逐 episode 帧数 assert 被注释；目前仅代码审查，尚未重新验证所有缓存帧与
数字行数一致，不能据代码合理就宣布完全排除缓存错位。

DP `_examples_to_dp_batch()` 取 `action_tensor[:horizon]`；底层推理从 `n_obs_steps-1` 开始
取动作。本次 n_obs_steps=1，因此起点是 0，没有发现这里额外增加的整帧 label shift。
这些是当前实现/配置检查，不是历史训练进程的逐样本运行时追踪。

## 另一个已定位的模型限制：预测被夹在 q01/q99 内

H20 DP 的 `VM4A/DiffusionPolicy.py:130` 写死 `DDPMScheduler(clip_sample=True)`。
数据归一化 `transform/state_action.py:135` 却允许 normalized 值到 [-2.2,2.2]。
两者范围不一致：训练可以看见 [-1,1] 外的标签，当前采样的预测实际限制在 [-1,1]。

已下载的 27k / 1019 episode0 所有右臂 prediction_chunks，反算 normalized 范围为
[-1.0, 1.000000067]。右关节6 q99 = -0.16190533 rad；GT 超过该值 0.01 rad 的 198 帧中，
非重叠 chunk 预测有 134 帧贴在该界限 ±0.005 rad 内。
这与原 GT 图的平顶对应，能解释部分动作幅度不足，不能单独解释所有毛刺，也不能解释 ACT 的全部问题。
DP 文件快照保存在本报告目录 `h20_DiffusionPolicy.py`。

H20 保存的 q01/q99 与混合训练集 56,726 帧独立重算相符（右臂最大差 6e-8）；
不是沿用旧 train17 统计。旧 GT 报告的 `fit_dataset: train17 only` 描述不可信。

## 下一步有区分力的实验

1. 同一 checkpoint/同一输入/同一扩散噪声，比较 clip_sample 开/关，测饱和帧与运动误差。
   先做独立评估，不改现有服务；这能便宜地验证已定位的范围限制。
2. 时间对齐 A/B：保持原图像、state、episode 划分、训练预算不变，只改变 action 最近邻/previous。
   统一边界范围，按正常运动、内部空档、夹爪跳变分别比较。只换数据后重新画 GT 不能证明模型效果改善。
3. 时间尺度 A/B 与第2项分开：保留原始时间参数作诊断基准，另一组在统一真实 30Hz 时钟上
   同时选图像、state、action。不能只换 action 时钟而保留原图像索引就称为完整重采样。
4. 验证训练 loader 缓存帧总数/每 episode 边界、逐样本 action/frame 配对与末尾 mask。
5. 真正留出集按完整原始轨迹互斥；`GT_1019_val_v1` 的 eab64427... 已在混合训练集，
   只能视为训练内诊断，不能作为泛化证据。

尚不能下结论“时间策略就是抖动根因”。不覆盖原数据，不提前认定 previous 正确，也不把
ACT loss 下降或图上平滑视作同步正确的充分证据。

## 后续已执行的验证（21:00—21:26）

### 1. H20 loader 与视频边界

脚本 `test/h20_loader_clock_audit.py` 在 H20 既有 Python 环境只读执行，未使用 GPU。
对 train/val 全部 456 路视频执行 ffprobe 全解码计数，均等于 metadata/Parquet 行数。
实际 `get_state_or_action` 方法在 912 个首帧、中间、末尾样本上返回的 action/state 与
真实 Parquet 一致；实际缓存 `get_video` 方法在 tiny sentinel 上的轨迹 offset/帧索引一致。

这排除了当前文件中“帧数短缺导致之后缓存整体错位”的直接证据；不是对历史训练进程
内存缓存的快照，也没有逐像素检验全量视频。报告为
`output/diagnostics/time_alignment_20261020_203853_185176/h20_loader_clock_audit.json`。

### 2. 现有预测 chunk 的时序一致性

`test/analyze_prediction_chunks.py` 验证：保存的执行曲线恰好等于每隔16帧取一个预测chunk
的正确拼接；同样拼接真实GT则与原GT完全相等，未发现绘图多移一帧的错误。

DP 的相同目标时刻在不同查询里的预测标准差平均 0.03266 rad；ACT 为0.02192 rad。
两者数据集/轨迹不同，不能将两数当成模型优劣排名。DP同时受不同输入和扩散噪声影响。
使用仅来自当前及过去查询的等权平均，DP二阶差分RMS从0.03205降至0.00671 rad/sample²，
ACT从0.03126降至0.00149。平均前后仍使用录制观察，不代表机器人闭环成功。
结果：`output/diagnostics/prediction_chunks_20261020_210033_512340/report.json`。

### 3. clip_sample 独立 A/B：关闭反而更差

`test/h20_dp_clip_ab.py` 使用同一27k checkpoint（SHA256 b49cf847...ff59）、同一eab64427...
训练内诊断轨迹，3个随机种子；每个on/off调用从相同RNG状态开始，重复基线误差为0。
只在该独立进程改变scheduler设置，未改训练代码、checkpoint或现有服务。

|3个seed平均|clip=True|clip=False|
|---|---:|---:|
|右臂MAE rad|0.03803|0.14759|
|二阶差分RMS rad/sample²|0.03171|0.25831|
|chunk边界平均步变 rad|0.04945|0.18559|

因此“裁剪导致平台”这个范围限制仍成立，但直接关闭裁剪不是当前checkpoint的修复方法。
不得把之前代码定位升级为“关掉即可解决”。结果为报告目录的 `h20_dp_clip_ab.json`。

### 4. 扩散噪声复用 A/B：只能解释部分抖动

`test/h20_dp_noise_ab.py` 保持clip=True，逐query单样本执行，3个seed对比每次换噪声与
跨query复用相同噪声序列。两组的输入、模型、批量大小均一致。

|3个seed平均|每query换噪声|跨query固定噪声|
|---|---:|---:|
|右臂MAE rad|0.03879|0.03652|
|二阶差分RMS rad/sample²|0.03226|0.02898|
|chunk边界平均步变 rad|0.05134|0.04418|

降低幅度有限，不能把扩散随机性认定为全部原因，更不能解释ACT。
结果为 `h20_dp_noise_ab.json`；两次GPU2诊断均已正常结束。

### 5. 将真实空档与预测毛刺在同一轨迹对上

`test/correlate_jitter_with_raw_gaps.py` 已验证 eab64427... 的GT右臂与混合训练集episode109
逐值完全相同，再由trim元数据还原原始头相机帧。该轨迹仅有一处内部action空档：
36.998—37.718秒，共21个保留帧。前后各扩展16帧后覆盖53/1376帧。

96.8%的预测/GT动作增量误差平方和在扩展区间之外；区间外预测平均步变0.01232 rad，
附近0.01174 rad，未见毛刺明显集中在空档附近。
区间外chunk边界平均步变0.05117 rad，非边界0.00978 rad；GT对应0.00480与0.00463 rad。
这将“本条轨迹的内部action空档是持续毛刺主要触发位置”的解释降为低优先级，不能排除
其他同步问题或全训练分布的影响。报告：
`output/diagnostics/jitter_gap_correlation_20261020_211404_761603/report.json`。

### 6. 找到原始DDS消息并对账184,419条

用户提供 `agi-192.168.168.197:~/app`。只读检查确认符号链接指向
`/data/app/a2d/backup/app`；`bin/dds_record`、`bin/dds_parse`、`bin/teleop_main_node`
是编译后的ELF；未执行程序或操作机器人。SDK协议提供header stamp/seq定义，
`conf/record/record_topic.json`包含 `/wbc/arm_command`，但未找到匹配H5写入逻辑的源码。
SDK标准字段编号与pbdat内实际字段编号不同，不能直接把SDK协议当作原始消息schema。

原始轨迹目录还保留 `record/wbc_arm_command/*.pbdat`。本地独立脚本
`test/audit_pbdat_action_clock.py` 解析记录封装和protobuf wire fields；不调用厂商二进制。
字段识别通过全部消息对账：152/152条轨迹、184,419条消息，时间戳、右臂position与H5
逐元素完全一致，14个关节名/顺序也全部一致。文件末尾、消息计数和消息长度均完整消费。
因此没有发现这些原始DDS记录转H5时再丢掉机械臂消息的证据。

最长空档所在2c97...：消息内stamp相邻差3440.492ms；pbdat封装时钟差3439.857ms；
恢复后的消息间隔约10ms，未观察到保存下来的积压消息集中补到。eab...对应两值720.266ms
和715.921ms，同样在原始消息层已存在空档。

重要限制：封装中4个整数时钟逐条相同，名称/阶段含义未知，不能称为独立发送/接收时钟。
部分其他轨迹还出现epoch级/uptime级数值切换，共15处，已从时钟间隔比较剔除，不能误报
为超大延迟。剔除后未见“封装间隔>100ms但消息stamp不间断”的事件；仅有1次间隔差>50ms。
消息header字段中没有已确认可用的逐命令序号，故不能区分发布端暂停、通信漏收与录制漏记。

当前证据支持“原始记录里本来就没有这段机械臂消息”，不支持“后续H5/LeRobot转换丢掉”
或“已保存了延迟补到的连续命令”。但不能据缺少记录证明上游从未发过消息。
最终报告：`output/diagnostics/pbdat_action_clock_20261020_212548_468638/summary.json`
和同目录 `episodes.jsonl`。

尚未完成的因果验证：采集时间戳赋值源码、发布暂停语义，以及严格控制变量的
nearest/previous/统一真实30Hz训练消融。当前证据不足以将某种策略宣布为最终正确方案。

## nearest / previous 配对短微调（后续已完成）

已在物理GPU3顺序完成两条件×3seed×600步，总计3600步；耗时257.5秒。
正式27k checkpoint、数据和服务均未修改。使用同一eab64427...训练内轨迹，
固定图像、state、夹爪、归一化、随机抽样和初始化，只替换右臂7维nearest/previous标签。
避开原始action覆盖区间外的窗口，两组共用1345个训练起点及1344帧评估范围。

下载结果后独立复核：三对起始预测数组逐元素完全相同，六组日志均到600步；最终数组
重算MAE与远端结果相符。该实验是短微调而非完整重训，不声称闭环验证或正式模型验收。

|三seed平均|nearest训练|previous训练|
|---|---:|---:|
|右臂MAE，对nearest标签，rad|0.02886164|0.02887354|
|右臂MAE，对previous标签，rad|0.02870437|0.02870007|
|二阶差分RMS，rad/sample²|0.02474556|0.02470838|
|chunk边界平均步变，rad|0.03494394|0.03485783|

previous边界跳变仅低约0.25%；两组曲线几乎重合，未观察到更换该策略消除毛刺。
两组都比微调前改善，不能将共同微调效果归因于previous。seed44两条件的末50步loss
均未低于首50步，不能宣称全部loss门禁通过；当前仅报告已实际观察到的预测指标。

这降低了“命令nearest/previous选择本身是主要抖动原因”的优先级，但仍未排除整帧级
同步偏移、真实30Hz重采样的影响、其他数据批次或完整重训后的不同结果。

入口：`test/h20_train_alignment_ab.py`；详细证据与图：
`output/diagnostics/alignment_train_ab_20261020/findings.md`。

## 图像内容与state的整帧同步线索（后续已完成）

入口 `test/audit_visual_state_timing.py`：原始H265/JPG逐帧运动与右臂state区间速度对照，
9条代表轨迹、18路相机、36,998帧。1019六条右腕的最佳lag为0,0,0,10,10,20ms，
CV R²为0.34～0.54；未发现这些样本普遍相差100～200ms的证据。其他批次部分拟合质量
很差，不能据此宣布全量物理同步通过。

头相机受人手/物体运动影响；固定右下ROI追加检查显示2c97...与eab...的平均峰值在
-60/-50ms附近，但五个时间块的估计不完全一致，峰值很宽，不能当作曝光延迟标定。
原始消息赋时语义、相机曝光时刻与state滤波仍需独立证据。

独立推理对照（同checkpoint/同state/同噪声，3seed）：正确图像右臂MAE0.03803 rad，
打乱图像0.07139，全黑0.13848，表明当前模型对图像有明显依赖。

仅将头图像平移-3～+3帧的21组扫描没有改善误差/毛刺；零偏移MAE0.038774 rad，
所有偏移均为0.03878～0.03890，chunk边界步变仍约0.0509 rad。
这不排除重训后或其他模态的对齐作用，不能用来证明某个物理偏移为零。

完整方法、失效边界、失败重跑记录和图表：
`output/diagnostics/visual_state_timing_20261020_215534_094873/findings.md`。

## 统一真实30Hz网格的全量只读对照

`test/audit_uniform_clock.py` 对152条轨迹还原原始trim范围，并按首个保留头图时刻建立
真实30Hz网格。只统计选图/数值，不重编码或替换数据。独立验证最近邻边界、等距选择、
理想时钟恒等映射和152条帧计数恒等式均通过。

同序号换头图的位置：0923为1984、1004为654、1015为6、1019为15。
action差异主要集中旧批次；更换图像索引不等于原数据帧内错配。

已分析DP GT的eab64427...，1376帧统一网格后头图序列完全不变，无复用/漏选；
按所选图像时刻取action与当前标签相同。若改在规则网格时刻取数值，action平均仅变
0.000216rad，同时会引入图像实际时刻与网格数值时刻的残差。

因此本条持续毛刺目前也缺乏“统一时间网格即可消除”的证据；旧批次的时间尺度和完整
重训影响未排除。不能只改Parquet timestamp而不处理图像选择，就宣称完成真实重采样。

详细证据：`output/diagnostics/uniform_clock_20261020_223847_384929/findings.md`。

## 同一时间位置的chunk切换对照

`test/audit_chunk_boundary_phase.py` 穷举16种chunk拼接相位，DP共同目标1344帧，ACT176帧。
同一个时间转换位置有一次切换query、15次不切换query，以消除真实运动强弱的混淆。
DP切换/不切换的平均步变分别0.051307/0.009866rad，ACT分别0.061025/0.002377rad。
99.85%的DP时间位置、100%的ACT时间位置切换时更大。

真实GT的16种相位拼接逐元素一致；预测边界只占约1/16时间，却承担DP68%～77%、
ACT89%～92%的增量误差能量。这是不同query预测不一致造成额外跳变的直接证据。
不是训练根因的唯一识别，也不等于平滑部署即可修复；闭环结果和完整训练消融仍未验证。

详细证据 `output/diagnostics/chunk_phase_20261020_224422_152991/findings.md`。

## 固定预测步偏差的跨时间段验证

`test/audit_horizon_bias.py` 将每条已有预测轨迹分前后半段，分界两侧剔除16帧，
在一半估计每预测步的关节偏差，在另一半测试，并做反向检查。
DP边界增量误差RMSE只变化约0.1%～0.3%，ACT校正反而恶化；未支持固定预测步偏差表
可解释/消除跳变。实际query误差中，块内共同偏移RMS比剩余形状误差更大，且不能由
跨时间段固定bias稳定解释。该GT辅助操作仅为诊断，未用于部署。

这缩小了可疑机制，但还不能区分观察歧义、真实时间偏移、训练目标与模型泛化。
证据：`output/diagnostics/horizon_bias_20261020_224736_358340/findings.md`。

## 固定时间偏移的跨时间段验证

`test/audit_fixed_time_shift.py` 在前/后半段拟合全chunk右臂MAE最优的整数偏移
（-8到+8帧），在另一半段验证；统一候选有效窗口并在分界剔除共享窗口。
正偏移表示 prediction[t] 与 GT[t+shift] 比较，仅是离线诊断。

DP两方向拟合偏移-2/-3帧，独立段MAE分别从0.044753降至0.043317、
0.033551降至0.031349rad（约3.2%/6.6%）；边界增量误差RMSE只改善约1%。
旧ACT轨迹拟合+5/+3帧，独立段MAE改善约17%/26%，但边界增量误差RMSE反而略升。
因此固定相位偏差可以解释部分位置误差，却不足以解释现有边界抖动。
这不排除采集对齐影响训练，也不能把帧偏移解释为实测传感器延迟。
ACT和DP来源、轨迹不同，不能用这里数值给模型排名；并非用户新提供的单轨迹9000步运行。

证据：`output/diagnostics/fixed_time_shift_20261020_225445_522929/report.json`。

## 用户单轨迹9000步图的实际数组复核

已定位H20 `DP_1019plus1015_mask_h32_ep018_overfit_10k_20261020`，train episode18共314帧。
右臂7关节整块执行MAE2.69°，优于同查询起点state保持16帧的4.03°，支持学到未来运动。
但每帧复制真实state为1.37°，每帧重新预测第一步为2.34°，不能仅凭2–3°证明预测链路充分拟合。
更关键的是这次块内平均步变0.04685、边界0.04749rad/sample，几乎相同：
此次强抖动并非仅由块切换造成，与此前样本机制不同。
配置明确horizon32、n_action_steps16，因此不是看到32/16就能认定错配。
详见 `experiments/gt_analysis/single_ep018_20261020/findings.md` 和其 `chunk_audit/report.json`。

后续检查：该轨迹无>100ms源action断层，但存在开头2/末尾23帧action覆盖外端点填充。
剔除后289帧的预测二阶差分RMS仍0.10560、GT0.00301，无法用边缘填充解释内部抖动。
同checkpoint三seed配对将DDPM推理20步增至100步，MAE0.04581→0.04629rad，
二阶差分RMS0.10175→0.09443（只下降约7.2%），没有解决问题。
实验只读使用空闲GPU1且已完成退出；数据和训练未修改。详见同目录findings后续章节。

单轨迹归一化复核发现明确范围不一致：q99训练变换允许[-2.2,2.2]，
DDPM clip_sample=True将输出限制在[-1,1]；右joint1/7的GT平台超出本次统计q99，
而预测不能达到该值。这解释部分平台偏低，但不可避免MAE下界只有0.00259rad，
且裁剪GT仍平滑（其二阶差分RMS0.00291，对比预测0.10367）。
因此不能用限幅单独解释主体抖动。定量结果见单轨迹目录 `bounds_audit.json`。

同一9000步checkpoint的EMA/训练权重三seed配对进一步检查：MAE0.045805/0.045266rad，
二阶差分RMS0.101746/0.097283，仅小幅变化；不支持EMA选择是主体原因。
源码训练loss为噪声预测与加权delta之和，不是自由采样后的关节角MAE。
结果见单轨迹目录 `weights_ab.json`，该独立GPU1实验已完成且未写回任何模型。

进一步代码核对：差分辅助项在epsilon还原的clean动作上计算，各噪声时间步共用0.053，
未做SNR缩放；实际scheduler的epsilon→clean误差增益t98约64、t99约2029。
9000步前最后1000次更新中，weighted_delta/base中位1.28、p95=47.29，61.8%超过1。
公式本身正确，但辅助目标不一定弱，属于训练消融候选，尚非抖动根因证明。
未测梯度范数，loss比例不能解释为梯度比例；详见单轨迹报告和loss_schedule_audit.json。

随后已补测固定噪声时间步的梯度分解（4真实查询×3seed）：加权差分/基础梯度
范数比值中位数t0/t50/t90/t99为0.000906/0.760/5.638/665.802。
这确认高噪声固定batch中差分项可以主导梯度；真实训练混合时间步，不能将此
直接作为训练整体占比或根因证明。实验无优化器步骤、无EMA更新，12组完成。
详见单轨迹目录 `gradient_audit.json`，此前“未测梯度”仅指日志汇总阶段。

差分项短续训消融现已完成：同9000步权重、三seed、每条件600步、同输入/抽样/RNG，
保留0.053与关闭0的MAE均值0.041763/0.041610rad，二阶差分RMS0.101075/0.100901。
关闭仅改善0.37%/0.17%，没有解决抖动。12个前后预测数组复算通过，配对初值逐元素一致。
梯度放大不能据此直接认定为主因；短warm-start也不等于从头训练因果消融。
详见单轨迹目录 `loss_train_ab/verified_summary.json`，独立GPU1任务已完成。

回到原始时间对齐：单轨迹覆盖内289帧对照最近邻/过去样本/线性插值，
相对最近邻标签MAE分别0/0.001057/0.000489rad；二阶差分RMS分别
0.003013/0.003422/0.002172，对比模型0.105599。最近邻与评估GT数值验证一致。
不支持最近邻直接在GT写入模型同量级毛刺；不排除训练放大小误差或物理时间偏移。
详见单轨迹目录 `alignment_policy_audit.json`；插值仅反事实，并非正式推荐转换策略。

训练输入内容与精度后续核验：单轨迹314帧三相机141797376像素值与GT评估完全一致，
registry只选episode18。归一化state存在float64/float32及打包float16路径差异，
但三seed配对推理改用真实训练state后，MAE0.045805156→0.045804650rad，
二阶差分RMS0.101746432→0.101742893，影响可忽略。详见单轨迹目录
`loader_content_audit.json`、`state_precision_ab.json`。原训练/转换代码未修改。

解析oracle去噪器进一步验证：保留实际DDPM20步、反归一化和16步拼接，用GT解析
epsilon替换网络后，三seed均还原平滑裁剪GT，MAE0.002589932、二阶差分RMS0.002908344，
与独立限幅GT指标在1e-7内一致。说明本轨迹当前强抖动并非采样/拼接链路必然产物，
需继续定位网络去噪拟合；不能据此排除训练时间对齐的间接影响。见单轨迹oracle_ab.json。

## 从头训练对齐消融的完整结果（2026-10-21）

两个训练seed×nearest/linear，四组各10000步已完成，独立GPU1任务正常退出。
40000步记录、exit_code=0、8份前后预测数组及配对初始预测逐元素一致均已核验。
两seed均值MAE0.0383044→0.0377213rad（linear低1.52%），二阶差分RMS
0.0879879→0.0858244（低2.46%）；seed42抖动升2.12%、seed43降7.53%。
双标签评分结论近似，没有发现稳定、大幅改善，不能将线性插值视为修复当前强抖动。
只检验单轨迹训练内的右臂动作重采样，不能排除真实采集延迟、其他观测时钟和长空档轨迹。
原转换、训练、评估代码与正式数据未修改；完整证据：
`experiments/gt_analysis/single_ep018_20261020/alignment_scratch/findings.md`。

## 重要新机制：零loss通道联合去噪（2026-10-21）

单轨迹checkpoint仍输出20维，但只监督右8维。独立诊断只对其余12维做解析去噪，
保持右8维原网络输出。三seed×baseline/未来GT辅助/当前state目标，共9组完成。
使用当前state目标（不使用未来标签）时右臂MAE0.045805→0.011292rad，
二阶差分RMS0.101746→0.006837，约下降75%/93%。效应明显大于本轨迹标签重采样消融。
强烈支持未监督通道推理处理是该模型的重要抖动机制，尚需跨模型/轨迹复核，未部署。
仅独立进程hook，不改原训练/转换/评估代码；详见单轨迹报告及masked_dims_state_ab.json。

后续持出集验证限制了结论范围：多轨迹27000步模型在val ep0/7/14上的三seed对照，
UUID训练集排除、18实际数组复算与2760次右8维保持断言通过，抖动只改善约5%～9%，
MAE一条略恶化、两条改善。未监督通道影响不能解释全部正式模型抖动，尚非通用修复。
详见 `docs/DP_MASKED_DIMENSIONS_DIAGNOSIS.md` 及单轨迹目录heldout_masked_dims/verified_summary.json。

## 8维动作结构训练验证完成

两seed各10000步，仅生成有监督的右8维、保留原20维state及原数据nearest标签。
相同独立训练协议下，20维right-mask基线/8维模型均值MAE0.0383044/0.0038163rad，
二阶差分RMS0.0879879/0.0038356，分别改善约90%/95.6%。两seed方向一致。
20000步日志和4份前后数组已独立复算，exit_code=0，未改原代码、正式模型或数据。
这是单轨迹拟合的结构验证，不是泛化或闭环通过；架构改变不声称相同初始权重。
与通道干预结合，强烈支持这条DP图的主要抖动来自20维联合生成/右8维监督不一致，
而不是nearest重采样本身。详见DP_MASKED_DIMENSIONS_DIAGNOSIS.md最新结果。
