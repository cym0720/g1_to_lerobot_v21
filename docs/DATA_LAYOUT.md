# 本机数据目录约定

原始数据与生成数据不进入 Git。原始记录保存在 `dataset/raw/`，按来源和批次区分；原始文件不修改、不删除。

```text
dataset/
  raw/<source>/<batch>/          # 原始录制和配套元数据
  converted/
    <data-standard>/              # 例如 eva-replay-g1-qpos20-v1
      <camera-standard>/          # wrist_lr 或 head_color_fisheye_lr
        <date-or-batch>/          # 单日转换基线
output/
  .scratch/                      # 可复核的临时工作数据
  .staging/                      # 未完成转换或临时打包数据
  current/<data-standard>/<camera-standard>/ # 当前流水线临时工作区，可重建
    data/                         # 中间数据
    reports/                      # 校验及 smoke 报告
  cleaned/<data-standard>/<camera-standard>/<batch>/ # 通过清洗门禁的批次数据
  merged/<data-standard>/<camera-standard>/<name>/   # 通过契约核对和门禁的跨批次数据
  diagnostics/<task>/<date>/     # 诊断与审计报告
  preflight/<task>/<date>/       # 原始数据预检输出
experiments/<experiment>/        # GT评估、模型实验、复现脚本和结果
artifacts/<kind>/                # 部署包、交付包和可移交资产
```

`dataset/` 只允许 `raw/` 和 `converted/` 两种阶段根目录。数据集目录先按**数据标准**，
再按**相机标准**分层。清洗后数据放在
`output/cleaned/<data-standard>/<camera-standard>/<batch-or-dataset>/`；跨批次数据放在
`output/merged/<data-standard>/<camera-standard>/<merged-name>/`。不得新建 `output/converted/`、
`output/cleaning/`、`dataset/cleaned/` 或 `dataset/merged/`。已有旧路径先核验并归档，
不能只因路径旧就删除或移动。原有 `output/cleaning/` 历史工作区已完整归档到
`artifacts/datasets/legacy_cleaning_20261029/`，保留原有阶段树供追溯。

符合当前契约、已完成验证且有完整 `meta/info.json` 和数据标准元数据的单日基线放在
`dataset/converted/<data-standard>/<camera-standard>/<date>/`。数据标准应能从所用契约
或明确版本化的关节布局识别；不能只凭目录名或 `meta/info.json` 中的旧 contract 标签认定，
还要核对 state/action 的具名维度和来源顺序。`head_color_fisheye_lr` 对应头部 RGB
及左右鱼眼视角；`wrist_lr` 对应头部 RGB 和左右腕部相机。鱼眼与腕部相机使用不同
的相机目录，不能把两种相机流混入同一个数据集；多日期拼接数据也不能放在
`dataset/converted/`，通过合并门禁的结果进入 `output/merged/<data-standard>/<camera-standard>/<name>/`，
未通过合并门禁的历史拼接或重建过程产物归入 `experiments/legacy_conversions/`。未完成或缺少完整元数据的转换留在
`output/.staging/`。模型 GT 对比、回放图、训练实验结果和实验脚本放在
`experiments/`，部署及交付包放在 `artifacts/`。`output/diagnostics/` 只保留
数据质量、统计和运行诊断；正式清洗数据仍按 `cleaned/`、`merged/` 阶段归档。

部分历史 wrist 数据使用旧版 `observations.state.qpos` / `action.qpos` 字段且没有
`meta/data_standard.json`。这些目录按相机键和单日来源归位，不代表其关节语义已通过
当前契约核验；正式训练前仍须验证字段顺序和转换来源。

历史相机优先目录已按现有元数据归档到新层级：鱼眼批次在
`g1_head_color_lr_fisheye_qpos20/head_color_fisheye_lr/`，wrist 1026/1027 在
`g1_wrist_lr_qpos20_v1/wrist_lr/`，EVA 声明的 wrist 1008/1015/1019/1020 在
`eva-replay-g1-qpos20-v1/wrist_lr/`。旧 wrist 0923 的 feature names 尾部标注
`[lift, waist_pitch]`，与其 `conversion_provenance.contract=eva-replay-g1-qpos20-v1`
声明的 `[waist_pitch, lift]` 相冲突，不能归入可信的 EVA 标准；对应旧转换产物待清理，
原始批次仍保存在 `dataset/raw/g1/0923_g1_simple_data/`。旧 1008 等批次的 EVA 标记表示
其输出声明，正式使用前仍应结合 Parquet 顺序和源批次映射复核。

通用流水线脚本默认把中间结果写入 `output/current/<contract_id>/<camera-standard>/`。它不是正式清洗数据集的归档位置：
按数据标准处理相机、执行 224×224 等契约要求，并完成清洗阶段的完整视频/Parquet 验证后，
再写入新的 `output/cleaned/<data-standard>/<camera-standard>/` 目标目录。跨批次合并另写入
`output/merged/<data-standard>/<camera-standard>/`，不得覆盖已有目录。

原始压缩包统一放在仓库外的
`D:\go work\archives\data-cleaning\raw-bundles\`。压缩包可能是唯一原始副本；删除前先确认
目标归档完整并核对文件哈希。不要因为找到同日期的解压目录就删除压缩包。

验证器日志、临时 LeRobot smoke 数据和缓存属于可再生内容，不应长期留在 `dataset/`。
