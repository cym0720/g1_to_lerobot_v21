# StarVLA G1 WSL 验证工作区

该目录是 `~/Projects_2026/starvla-g1-validation` 的可复现模板。它与 Windows
侧的数据清洗仓库分离：这里仅运行 StarVLA 的训练、推理和环境验证。

## 目录职责

```text
starvla-g1-validation/
├─ scripts/                 # G1 ACT 训练、推理、服务脚本
├─ docs/                    # StarVLA 使用说明
├─ third_party/starvla/     # 上游 StarVLA Git checkout
├─ requirements.txt         # 本工作区唯一的安装入口
├─ setup_uv.sh              # 用 uv 创建环境并安装依赖
└─ validate_environment.py  # CUDA、关键依赖与 StarVLA 导入检查
```

训练产物不写进该 Git 工作区。请使用 WSL 原生路径，例如
`~/artifacts/starvla/runs`；训练数据初期可只读访问 `/mnt/d/...`，长时间训练时再
复制经过验证的数据副本到 `~/datasets/g1/`。

## 首次使用

```bash
./setup_uv.sh
.venv/bin/python validate_environment.py
```

脚本默认使用 Python 3.10 与 CUDA 12.4 的 PyTorch 轮子。CUDA 轮子单独由
`requirements-cuda124.txt` 安装，其余依赖仍只继承上游清单，避免多索引解析冲突。若机器的
CUDA 驱动不兼容，不要直接修改上游依赖；先在本工作区的 issue/说明中记录目标版本后再调整。

依赖默认从清华 PyPI 镜像下载，CUDA PyTorch 额外使用官方 CUDA 索引。可通过
`PYPI_INDEX_URL` 或 `PYTORCH_CUDA_INDEX_URL` 临时覆盖源地址。
