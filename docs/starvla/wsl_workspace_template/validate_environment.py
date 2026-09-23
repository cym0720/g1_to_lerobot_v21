"""检查 StarVLA G1 验证工作区的 Python、CUDA 和关键模块。"""

from __future__ import annotations

import importlib
import json
import platform
import sys


REQUIRED_MODULES = ("torch", "accelerate", "transformers", "starVLA")


def main() -> None:
    """输出机器可读结果；任一关键依赖缺失时以非零状态退出。"""
    missing = []
    versions = {}
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            missing.append(module_name)

    torch = importlib.import_module("torch") if "torch" not in missing else None
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": versions,
        "missing_modules": missing,
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch else None,
        "gpu_count": torch.cuda.device_count()
        if torch and torch.cuda.is_available()
        else 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit("关键依赖缺失：" + ", ".join(missing))


if __name__ == "__main__":
    main()
