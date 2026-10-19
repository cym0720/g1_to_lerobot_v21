# H20 训练与部署快照

这里保存针对 H20 远端环境的历史实验和部署脚本，不属于数据清洗主流程。
脚本含特定数据批次、GPU 分配、远端绝对路径和外部 StarVLA 依赖；不要在本机直接批量运行。

- `train_single_gpu_1008.py`：原根目录训练入口，保留 1008 单进程检查。
- `staging/train_single_gpu.py`：原 `scripts_stage/` 里的后续版本，额外支持 1015 训练及双进程检查。
- `staging/`：下载、注册、补丁、配置与启动脚本的原部署快照。
- `run_baseline_formal.sh`：H20 固定工作区的 ACT/DP 启动器。

两个训练入口存在实际行为差异，因此保留为不同快照，不能当作重复文件直接删除。
训练文件的 `__file__.parents[2]` 按远端部署位置计算工作区；本目录仅保存源码，
部署时仍需放回启动器所引用的远端 `baselines/experiment/1008_only_10k/train_single_gpu.py`。
本次目录整理没有修改或启动远端作业。

评估检查脚本位于 `tools/diagnostics/h20/`，下载脚本位于 `tools/transfer/h20/`。
