# 两组右臂 action8 DP 实验

> 历史实验记录：文中 `test/` 下的一次性验证脚本已于 2026-10-29 清理；本文件记录已完成实验的结果，不是当前可运行流程说明。

## 最新执行状态（取代下方旧的待审批记录）

用户已批准第二组启动，并指定只用原5条中剩余3条验证。第二组采用联合清单直接读取
两份已有清洗数据，不复制或改写源数据：train41条26125帧，val3条703帧。
排除1b88c505与eab64427两条，不加入训练；共同对比使用68fc6cdf、7558876a、7ab2aa22。
新清单`experiments/h20/dp_right8_20k_preparation/1015_1019_41_val3_plan.json`。
第一组保持135训练/17留出，自动验证仍输出原共同5条，可筛选其中3条公平比较。

第二组H20 GPU2进程2261947已实际更新参数，完成500步；首步更新2.01166e-7，
最近100步loss0.4150。全部44条三路视频和数值已读取通过。
输出`/data2/cym_ws/DP_right8_20k_20261021/clean41_val3`，日志同级`clean41_val3.log`。
第一组GPU1已到14800步。两组均20000步目标，尚未完成。
第一组后台watcher已确认5k和10k的独立复算及右臂GT/state绘图通过；
日志`full135_validation_watch.log`。第二组每次保存checkpoint后内置复算和绘图。
所有进程均nohup在H20运行，关闭本地电脑不影响训练及验证产物生成。

第一组15k验证已独立复算。对共同5条2328帧按帧数加权：

|step|右7 MAE(度)|copy state MAE(度)|预测二阶差分RMS|边界delta-error RMS|块内delta-error RMS|
|---:|---:|---:|---:|---:|---:|
|5000|6.373|2.623|0.05362|0.14837|0.00950|
|10000|4.044|2.623|0.03807|0.10428|0.00765|
|15000|3.281|2.623|0.03300|0.09003|0.00701|

前5k到15k的验证误差及边界抖动连续下降，但仍高于copy-state MAE基线。
预测二阶差分也高于GT约0.002左右的量级；当前有改善趋势，尚非最终效果结论。
每一阶段具体曲线和数组在H20相应`validation/step_*`目录；本地已取回5/10/15k。

## 两组训练已完成并按共同3条作最终比较

2026-10-21 H20退出码均0。第一组20k checkpoint SHA256：
`14cbeefde546077b55ca684af7ee23a4f4edcc064ce75153ae236f46ff4310d0`。
第二组4个checkpoint和三条验证全部完整；第一组watcher确认4个checkpoint验证/绘图完成。
两组在相同query frame、固定DDPM推理种子和三条完整轨迹上评估，合计703帧。
右7 MAE/预测二阶差分/边界delta-error如下：

|step|全量135 MAE°|清洗41 MAE°|全量抖动RMS|清洗抖动RMS|全量边界误差|清洗边界误差|
|---:|---:|---:|---:|---:|---:|---:|
|5000|4.259|3.270|0.04811|0.04182|0.13260|0.11558|
|10000|2.430|2.098|0.02853|0.02740|0.07774|0.07443|
|15000|1.921|1.827|0.02503|0.02401|0.06807|0.06529|
|20000|1.728|1.806|0.02312|0.02347|0.06320|0.06383|

copy-state基线右7 MAE为2.457°（恒定）。两组在10/15/20k均低于基线；
预测抖动和边界误差从5k到20k持续下降。20k两组差异很小，第一组MAE略优，
第二组的更低误差优势没有稳定保留到最后一个checkpoint。分块边界误差仍约0.063rad，
高于块内约0.0055rad，推理stride16的边界效应尚未消失。

三条验证上夹爪MAE：全量组0.22/0.05/0.03/0.03，清洗组0.13/0.06/0.06/0.06
（依次5/10/15/20k；fraction单位）。2万步单条right_joint_3仍可见小跳变，
所以“指标单调改善”不等于动作完全无抖动。离线验证不等于真机闭环。

全部验证NPZ、independently_verified.json及right-arm PNG已取回至
`experiments/h20/dp_right8_20k_preparation/{full135_validation,clean41_val3_validation}/step_*`。
例如三条共同验证之一：
`experiments/h20/dp_right8_20k_preparation/full135_validation/step_020000/68fc6cdf-051b-42c9-bd70-edb9fc83ad46.png`。

## 固定实验条件

每组从头训练20000步，输入state20和三路224×224 letterbox图像，输出右7关节加右夹爪。
action源索引为8至15；不是只输入右臂state8。horizon32，执行16步，batch24，seed42。
已有图像不重复缩放，不修改原转换、训练或评估代码。各模型仅用自身训练集拟合统计量。
checkpoint每5000步保存并对共同的5条完整留出轨迹做离线预测；指标包含右7关节MAE、
二阶差分RMS及夹爪MAE。另有12条完整轨迹在第一组中保持留出，后续单独评估。
离线验证不能替代真机效果确认。

## 第一组：全量数据排除共同验证轨迹

原152条中的135条训练（55101帧），17条留出（4936帧）。
为了与第二组公平比较，把原全量train中的两个共同验证UUID移出训练；不改原文件。
独立清单：`experiments/h20/dp_right8_20k_preparation/full_135_plan.json`。
首次读取预检已通过135训练+5共同验证的全部三路视频、数值及帧数检查，退出码0。

远端根目录：`/data2/cym_ws/DP_right8_20k_20261021`。
首次运行`full135_run`在首步更新检查读取空参数时退出，exit_code=1；未产出checkpoint。
原始失败日志保留`full135_training.log`。独立脚本已改为检查有梯度的非空参数，
重启输出`full135_retry1`、日志`full135_retry1.log`、退出码`full135_retry1_exit_code.txt`。
运行脚本为`/data2/cym_ws/diagnostics_alignment_scratch_20261021_v1/h20_train_right8_manifest_v3.py`。
本段记录启动事实，不代表训练完成；须检查日志、有效参数更新、最终checkpoint和退出码。

重启后已核验首步：loss=1.155076、梯度范数9.583745，非空参数
`obs_encoder.key_model_map.cam_high_rgb.conv1.weight`最大更新2.01166e-7。
日志已到1400/20000步，最近100步平均loss=0.152742；GPU进程3750805占5270MiB。
这证明训练发生有效更新，不证明验证或闭环效果；5000步验证尚待产出。

### 5000步验证已产出（取代上述待产出状态）

checkpoint SHA256：`92fed6baf49fdb5b3e19b05493fbb925b12e0c06eb623f89f71419383c0e1b25`。
5条2328帧预测已取回，本地目录`experiments/h20/dp_right8_20k_preparation/step_005000`。
独立脚本`test/verify_right8_validation.py`核验UUID隔离、全部数组形状/有限性、query索引，
复算原报告三项指标，exit=0；另计算16步保持当前state的基线和边界/块内变化误差。

|UUID前缀|右7 MAE(度)|保持当前state MAE(度)|预测二阶差分RMS|GT二阶差分RMS|
|---|---:|---:|---:|---:|
|1b88c505|4.169|2.485|0.04556|0.00312|
|68fc6cdf|4.222|2.444|0.04736|0.00249|
|7558876a|4.827|2.571|0.05225|0.00295|
|7ab2aa22|3.753|2.360|0.04488|0.00191|
|eab64427|7.852|2.733|0.05789|0.00221|

分块交界处delta-error RMS为0.124–0.160rad，块内为0.00776–0.01016rad。
此时全部5条MAE均劣于基线，强边界抖动仍在；夹爪MAE为0.205–0.420。
不能把训练loss下降或单轨迹action8改善解释为批量泛化已通过。
这是5000步中间结果；继续训练至20000步并按相同口径复核后续checkpoint。

## 第二组：待用户检查的1015+1019合并方案

源数据直接使用清洗后的完整split：

- `artifacts/datasets/legacy_cleaning_20261029/1015/training_split_trim_letterbox224_v1`：train36条11191帧，val4条952帧。
- `artifacts/datasets/legacy_cleaning_20261029/1019_training_split_trim_letterbox224_v1`：train5条14934帧，val1条1376帧。

按原split合并，得到train41条26125帧、val5条2328帧。按original_episode_id排除重复及
train/val交叉；保留完整轨迹，不随机拆帧。原始action/state及图像保持不变，不重新裁剪、
平滑、对齐或缩放。只在新目录重建episode/frame/global索引、视频路径和元数据；
训练统计量只从最终41条train计算，val使用同一份train统计量。

已完成只读预检：46条、138路视频尺寸/帧数正确；与全量数据同UUID的state/action逐值一致。
证据：`experiments/h20/dp_right8_20k_preparation/cleaning_sources_preflight.json`。
预备清单：`experiments/h20/dp_right8_20k_preparation/1015_1019_41_plan.json`。
合并后还需核验全部Parquet值、视频解码、索引连续性、UUID隔离和统计量来源。
ACT检查按用户要求跳过。第二组尚未执行合并或训练，等待用户检查方案。

## 时间对齐结论的边界

配对四组单轨迹实验共40000步，插值相对最近邻平均抖动只降低2.46%，两个seed方向不一致。
真正action8结构的两个单轨迹实验在保留nearest数据时显著改善，见
`docs/DP_MASKED_DIMENSIONS_DIAGNOSIS.md`。当前批量实验用于检查泛化是否保留改善，
不据此否认原始action空档或宣称物理时间同步已经解决。
