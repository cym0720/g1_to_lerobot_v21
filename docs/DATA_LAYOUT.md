# 本机数据目录约定

`dataset/` 不提交到 Git，按数据性质而不是历史日期堆放文件。

```text
dataset/
  raw/
    g1/          # 本机 G1 原始录制：HDF5、相机流、meta_info.json
    auxiliary/   # 与原始录制配套的 AgiBot / Robocoin 元数据
    sample_dataset/  # 小型原始格式样例
  converted/    # 当前契约下可重建的 LeRobot 转换基线；默认应为空
```

各批次名称（如 `0908_v3`、`618data`）保持原样，避免重命名 episode ID 或修改原始文件。
转换程序的输入应始终来自 `dataset/raw/`；生成数据先写入 `dataset/converted/<run_name>/`，再由
流水线物化到 `output/current/`。

原始压缩包不属于工作输入目录，统一归档到
`D:\go work\archives\data-cleaning\raw-bundles\`。它们可能包含尚未解压的轨迹，不能因为存在
同日期的展开目录而直接删除。

`dlb.db`、验证器日志、临时 LeRobot smoke 数据属于可再生内容，不应长期留在 `dataset/`。
