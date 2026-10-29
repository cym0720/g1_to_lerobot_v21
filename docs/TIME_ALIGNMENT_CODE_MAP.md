# 时间对齐代码阅读入口

此文件仅说明当前实现与已验证边界，不改转换策略。完整实验见
`TIME_ALIGNMENT_REVIEW_20261020.md`，单轨迹细节在
`experiments/gt_analysis/single_ep018_20261020/findings.md`。

## 当前转换顺序

1. `tools/conversion/00_convert_618_to_lerobot.py:383`：读取头相机文本的原始纳秒时间戳为target。
2. 同文件`:66`：H5每路使用自身timestamp，对每个target查最近样本，再组装state/action。
3. 同文件`:56`：`searchsorted`找左右候选；距离相等取左，超出两端夹到端点。
4. 同文件`:131`：只有配置显式选择`leading_state_trailing_hold`时，开头覆盖外arm action
   改用state；末尾仍保持最后命令。不能把配置可用等同于所有历史数据已采用。
5. 三路图像对target对齐后输出等长帧。FFmpeg路径的索引函数为`packages/ffmpeg_io.py:94`。
6. 同转换文件`:393`、`:440`：输出timestamp重新写为`frame_index/fps`，因此最终等间隔时间列
   不能反映原始时钟抖动或action空档。核查必须回到原始timestamp与裁剪meta。

最近邻的核心语义：

```python
right = searchsorted(source_times, image_time)
left = right - 1
index = argmin(abs(source_times[left_or_right] - image_time))
```

它可能取图像之后的命令；没有最大时间差拒绝条件。缺口中会保持左/右两端样本，
在中点附近换端，图看上去连续不代表原始action持续发布。

## 哪些结论已有证据

- 最终152条轨迹数值回查与该最近邻实现一致，没有发现字段错排或落盘数值损坏。
- 原始action空档确实存在；raw DDS与H5逐记录匹配，不能归咎于LeRobot转换丢掉这些记录。
- 采集上游是否暂停发送、传输是否丢失、各时间戳代表何时，仍缺可靠源码/日志证据。
- 当前单轨迹episode18没有>100ms的action空档，标签本身平滑，模型强抖动远大于标签。
- 头图时间轴重新写成30Hz存在信息损失；这与“action取过去还是最近”是两个独立问题。

## 尚不能据此决定的修改

过去样本取法符合命令保持的一种解释，但只有时间戳语义成立才是物理正确对齐。
线性插值能减少标签离散抖动，但未必符合离散命令执行语义。不要凭曲线更平滑直接替换。
state晚于action还包含执行器响应，不能仅用最佳相关lag将训练标签整体移动。

episode18最近邻/线性插值的两seed配对实验已完成：四组各10000步，
初始预测配对完全一致，40000步日志及8份前后数组已核验，并分别对两套GT评分。
线性插值使两seed平均MAE降低1.52%、二阶差分RMS降低2.46%；
单seed抖动分别增加2.12%和降低7.53%，未显示稳定的大幅改善。
因此在这条轨迹上，最近邻重采样不是强抖动的主要解释。
核验与限制见`experiments/gt_analysis/single_ep018_20261020/alignment_scratch/findings.md`
及同目录`verified_dual_label_scores.json`。这不能代替物理同步校准或泛化验证。

后续真正action8结构的两seed训练内实验显著改善，但批量验证仍在进行；
证据见`docs/DP_MASKED_DIMENSIONS_DIAGNOSIS.md`，不能直接外推真实闭环效果。
