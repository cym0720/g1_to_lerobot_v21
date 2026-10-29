# DP右臂mask与联合去噪诊断

> 历史实验记录：文中 `test/` 下的一次性诊断脚本已于 2026-10-29 清理；结论和已保存的实验产物仍作为历史证据保留。

当前证据支持：右8维受监督、20维联合反向扩散时，未监督12维的轨迹会影响受监督输出。
这是已有模型的离线机制证据，不是已完成的正式修复或闭环验收。

## 已复核结果

单轨迹9000步checkpoint，三seed只对未监督12维使用当前state为固定clean目标，
右8维epsilon保留网络输出，MAE0.045805→0.011292rad，二阶差分RMS0.101746→0.006837。
9份完整预测独立复算通过；2440次hook干预断言右8维输出逐元素未变。
没有对右臂输出做平滑、没有向右8维写入GT/state，没有使用未来标签。

多轨迹27000步checkpoint，另一条训练内轨迹1376帧、三seed同类验证：
MAE0.038790→0.033272rad，二阶差分RMS0.032260→0.025142；块内改善更明显，
边界仍跳变，不能称作所有抖动已解决。ACT尚未作此机制验证。

ACT实现边界已只读核对：当前环境实际加载
`/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/.envs/task_site_packages/lerobot/policies/act/modeling_act.py`。
predict_action_chunk一次调用self.model，末端`actions=self.action_head(decoder_out)`，
没有把生成的20维action反复送回DDPM去噪的路径。eval分支VAE latent为全零。
因此DP的“未监督生成通道污染迭代采样输入”不能直接作为ACT抖动解释，也不能靠
修改ACT输出里其他12维来修复已生成的右8维。共享特征和训练mask仍可能影响ACT训练，
但属于另一种待验证机制。源码哈希和精确行号见单轨迹目录act_mechanism_inspection.json。

### 单轨迹通道分组对照

三seed×4条件共12组，使用当前state目标、没有未来标签，右8维输出3660次断言不变，
12份完整预测数组独立复算通过：

|被约束的零loss通道|右7关节MAE rad|二阶差分RMS|
|---|---:|---:|
|无，原推理|0.045805|0.101746|
|仅左臂/左夹爪0..7|0.021672|0.027533|
|仅头腰16..19|0.042304|0.088814|
|全部12维|0.011292|0.006837|

左侧8维的干扰更显著，头腰通道也有影响，合并约束改善最大。维度间有交互，不能把
分组改善相加或当作独立贡献比例。这是干扰通道归因，不是画左臂，也不推断左臂原数据错误。
结果`experiments/gt_analysis/single_ep018_20261020/masked_groups/verified_summary.json`。

### 未参与训练轨迹：改善幅度有限

27000步多轨迹模型，最终val episode0/7/14（230/244/224帧），三seed×两条件。
逐条验证UUID不在train，18份实际预测数组全部复算，2760次右8维保持断言通过。

|val episode|原MAE→当前state约束MAE rad|原抖动→约束抖动|
|---|---|---|
|0|0.043017→0.043108|0.033941→0.032122|
|7|0.036011→0.034781|0.031427→0.029245|
|14|0.043037→0.040012|0.032976→0.030113|

持出数据抖动仅改善约5.4%/6.9%/8.7%，MAE有一条略恶化，说明单轨迹模型的巨大
改善不能推广为正式模型的通用修复；未监督通道是已证实影响因素，但正式模型还有
更明显的跨query/分块一致性与泛化问题。这里没有闭环验证，也没有修改正式代码。
完整证据`experiments/gt_analysis/single_ep018_20261020/heldout_masked_dims/verified_summary.json`。

### 持出集剩余误差：边界占主导，且局部恶化

从18份数组进一步分解，处理零loss通道后三条轨迹的块内增量误差RMSE明显降低，
但边界增量误差RMSE分别0.08288→0.08997、0.07244→0.08014、0.08250→0.08762rad，
三条都增大。约每16帧一次的边界承担90.1%～93.0%的增量误差能量（原70.3%～77.1%）。
因此整体二阶差分下降不能解释为所有控制跳变改善，当前state约束不能直接作为部署修复。
这是固定chunk相位的关联分解，尚非训练根因的完整识别。
val ep0没有原始action>100ms缺口，仍有该现象；另两条各有一个保留缺口，未在此自动归因。
证据同目录`residual_decomposition.json`，脚本`test/audit_masked_heldout_residuals.py`。

跨三个推理seed的有限样本分解进一步显示：约束零loss维度后的边界增量误差能量中，
种子间方差占18.5%～29.7%；三次预测均值仍有0.0679～0.0808rad边界增量误差RMSE，
不能简单认为再平均几次采样就可消除问题。均值仍包含有限样本噪声，不能把剩余
70%～82%当成严格确定性偏差比例，更不能直接解释为时间延迟。
结果`heldout_masked_dims/seed_boundary_decomposition.json`，恒等式已逐项验证。

## 代码与机制

H20项目：`/data2/cym_ws/pick_apple_1008_act_dp_groot_20261014/baselines/starvla_local`。
单轨迹run的config.full.yaml中action_dim/state_dim=20，horizon32、n_action_steps16，
loss_dim_weights仅索引8..15为1。模型输入动作仍为20维。

`starVLA/model/framework/VM4A/DiffusionPolicy.py`构造20维shape_meta，传递loss_dim_weights，
内部normalizer为identity，predict_action调用EMA policy。
`.../_dp_vendor/diffusion_policy/policy/diffusion_unet_image_policy.py`：

当前核对入口：94行初始化20维噪声，112行scheduler更新，223/239行condition/loss mask，
261/268行只在loss上应用维度权重。

1. compute_loss的trajectory为完整20维动作，先对所有维度加噪。
2. dim_w只乘到噪声MSE上，不改变UNet的20维输入，也没有训练未监督维度的预测能力。
3. conditional_sample从完整20维高斯噪声出发，UNet和scheduler每步更新完整轨迹。
4. 各动作维度经UNet通道混合，未监督维度推理轨迹不再保证接近训练时真实动作加噪分布。

诊断hook将未监督维epsilon替换为由当前state目标解析得到的epsilon，仍沿原scheduler更新，
避免直接把clean常数硬塞进所有噪声时间步。这个差别很重要：直接每步写clean值并非同一实验。

## 后续修正方向及边界

8维结构验证已启动（运行中，尚无结论）：独立脚本
`test/h20_single_ep_right8_scratch.py`，H20输出目录
`/data2/cym_ws/diagnostics_right8_scratch_20261021_v1`，Python PID489362、wrapper489361。
seed42/43各10000步，使用与此前nearest从头训练相同的episode18覆盖内anchors、
三相机、20维state、右8维标签归一化、batch24、lr/warmup/cosine、delta0.053和clip。
只生成8维动作，loss权重全1；评估将右8维按原统计反归一化，其他填充值不作为控制输出。
比较对象为已完成20维right-mask模型的同seed预测，结构与随机张量形状不同，
不能声明初始权重/随机轨迹逐元素配对相同。只有两seed训练内结果，不是正式模型修复。
新诊断目录会保存最终权重/配置供后续复核，未替换任何正式模型。
检查完成要求：exit_code=0、summary两组完成、20000步日志完整、实际数组复算。
当前仅验证首组100步正常，未完成。独立进程nohup运行，SSH断连不影响训练。

### 8维结构训练已全部完成（取代上述运行中状态）

seed42/43各10000步完成，exit_code=0，Python进程已退出；20000步日志、4份前后数组
和两组最终预测全部独立复算。诊断权重各约1.1GB仅保存在新远端目录，未替换正式模型。
以下只比较同一训练内轨迹256帧、同一原始nearest标签和评估帧：

|训练seed|20维仅监督右8维 MAE→8维 MAE(rad)|二阶差分RMS 20维→8维|
|---|---|---|
|42|0.0423204→0.0038419|0.0925090→0.0038343|
|43|0.0342884→0.0037907|0.0834667→0.0038368|

两seed均值MAE0.0383044→0.0038163rad（约2.19°→0.219°，降90.0%），
二阶差分RMS0.0879879→0.0038356（降95.6%），边界平均步变0.04769→0.00964。
保留原数据和nearest对齐、仅消除未监督生成通道的结构后，两seed均大幅改善，
结合已有通道干预，是这条单轨迹DP抖动主要来自生成/监督维度不一致的更强证据。
因此不应把该图主要归因为原始数据或最近邻重采样。

限制：架构和随机张量形状改变，不是逐参数同初始化的单变量实验；只有两seed训练内
拟合，不证明整批数据泛化或真实闭环。8维模型不是现有20维checkpoint的直接替代品。
评估数组另外12维仅是反归一化占位值，不可用于机器人控制。原代码与正式数据未改。
独立核验：`experiments/gt_analysis/single_ep018_20261020/right8_scratch/verified_comparison.json`。

诊断checkpoint重载也已验证：两份权重均strict=True加载，重新推理的256帧20维
存储数组（右8维真实预测、其余为占位）与训练结束时数组逐元素完全一致，最大差0。
SHA256分别18a3ea51...1e88b、f5df4cd5...c773，完整哈希在right8_scratch/reload_verified.json。
这确认已保存权重可复现上述结果；没有扩大为泛化/闭环通过。

- 从头训练真正8维动作模型，再按明确机器人契约组装20维控制量，结构上消除无监督通道。
- 或保持20维并给所有生成通道有定义的监督；这是损失/任务改变，需要独立训练验证。
- 对已有checkpoint，可把未控制维度作为已知条件，使用与噪声日程一致的约束方式；
  当前hook只是候选原型，尚需定义每一维保持目标、精确时间语义与任务兼容性。

当前state中的非右臂值不总能直接当action命令：夹爪单位、头部、腰部顺序和移动行为都要
按机器人契约核验。因此没有把诊断hook接入正式评估或控制，也没有改原代码。
不能用此机制消除已经确认的原始action空档与采集时间语义不明问题。

## 可复查文件

- `experiments/gt_analysis/single_ep018_20261020/masked_dims_arrays/verified_summary.json`
- 同目录9份npz、hook_audit.json、masked_export.log和seed42_right_arm_masked_dims_ab.png
- `experiments/gt_analysis/single_ep018_20261020/formal_masked_dims_state_ab.json`
- 独立脚本`test/h20_single_ep_denoise_ab.py --masked-oracle-ab --save-work 新目录`
- 总时间对齐证据`docs/TIME_ALIGNMENT_REVIEW_20261020.md`
