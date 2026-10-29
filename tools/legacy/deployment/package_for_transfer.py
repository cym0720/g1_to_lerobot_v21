"""
STAR-VLA ACT 打包脚本
用于将模型和脚本打包以便传输到另一台电脑
"""
import os
import shutil
import zipfile
import json
from pathlib import Path
from datetime import datetime

def create_package():
    """创建打包文件"""
    print("开始打包 STAR-VLA ACT 模型和脚本...")
    
    # 设置目录
    base_dir = Path.cwd()
    package_dir = base_dir / "starvla_act_deployment"
    zip_file = base_dir / "starvla_act_deployment.zip"
    
    # 清理旧的打包目录
    if package_dir.exists():
        print("清理旧的打包目录...")
        shutil.rmtree(package_dir)
    
    if zip_file.exists():
        zip_file.unlink()
    
    # 创建打包目录结构
    print("创建目录结构...")
    package_dir.mkdir(exist_ok=True)
    (package_dir / "starvla_act_scripts").mkdir(exist_ok=True)
    (package_dir / "output" / "starvla_act_g1").mkdir(parents=True, exist_ok=True)
    (package_dir / "external").mkdir(exist_ok=True)
    
    # 1. 复制脚本文件
    print("复制脚本文件...")
    scripts_src = base_dir / "starvla_act_scripts"
    if scripts_src.exists():
        for item in scripts_src.iterdir():
            if item.is_file():
                shutil.copy2(item, package_dir / "starvla_act_scripts" / item.name)
            elif item.is_dir():
                shutil.copytree(item, package_dir / "starvla_act_scripts" / item.name)
    
    # 2. 复制模型文件 (只复制 final_model 目录)
    print("复制模型文件...")
    model_tasks = ["g1_act_task_04", "g1_act_task_05", "g1_act_task_06"]
    for task in model_tasks:
        source_path = base_dir / "output" / "starvla_act_g1" / task / "final_model"
        if source_path.exists():
            dest_path = package_dir / "output" / "starvla_act_g1" / task / "final_model"
            dest_path.mkdir(parents=True, exist_ok=True)
            for file in source_path.iterdir():
                if file.is_file():
                    shutil.copy2(file, dest_path / file.name)
            print(f"  已复制: {task}")
        else:
            print(f"  警告: {task} 模型不存在")
    
    # 3. 复制框架代码
    print("复制 STAR-VLA 框架代码...")
    framework_src = base_dir / "external" / "starVLA"
    if framework_src.exists():
        shutil.copytree(framework_src, package_dir / "external" / "starVLA")
    
    # 4. 复制文档
    print("复制文档...")
    doc_file = base_dir / "DEPLOYMENT_GUIDE.md"
    if doc_file.exists():
        shutil.copy2(doc_file, package_dir / "DEPLOYMENT_GUIDE.md")
    
    # 5. 创建启动脚本
    print("创建启动脚本...")
    
    # Linux 启动脚本
    launch_sh = '''#!/bin/bash
# STAR-VLA ACT 启动脚本

echo "STAR-VLA ACT 部署环境"
echo "======================"

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 检查模型文件
if [ ! -d "output/starvla_act_g1/g1_act_task_04/final_model" ]; then
    echo "错误: 模型文件不存在"
    echo "请确保已解压所有文件"
    exit 1
fi

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)/external/starVLA

echo "环境设置完成！"
echo ""
echo "可用命令:"
echo "  1. 测试推理: python starvla_act_scripts/inference_starvla_act.py --help"
echo "  2. 启动服务器: python starvla_act_scripts/server_starvla_act.py --help"
echo "  3. 查看文档: cat DEPLOYMENT_GUIDE.md"
echo ""
echo "快速开始:"
echo "  python starvla_act_scripts/server_starvla_act.py \\\\"
echo "    --ckpt_path output/starvla_act_g1/g1_act_task_04/final_model \\\\"
echo "    --host 0.0.0.0 \\\\"
echo "    --port 5000"
'''
    
    with open(package_dir / "launch.sh", "w", encoding="utf-8") as f:
        f.write(launch_sh)
    
    # Windows 启动脚本
    launch_bat = '''@echo off
REM STAR-VLA ACT 启动脚本 (Windows)

echo STAR-VLA ACT 部署环境
echo ======================

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python
    pause
    exit /b 1
)

REM 检查模型文件
if not exist "output\starvla_act_g1\g1_act_task_04\final_model" (
    echo 错误: 模型文件不存在
    echo 请确保已解压所有文件
    pause
    exit /b 1
)

REM 设置环境变量
set PYTHONPATH=%PYTHONPATH%;%cd%\external\starVLA

echo 环境设置完成！
echo.
echo 可用命令:
echo   1. 测试推理: python starvla_act_scripts\inference_starvla_act.py --help
echo   2. 启动服务器: python starvla_act_scripts\server_starvla_act.py --help
echo   3. 查看文档: type DEPLOYMENT_GUIDE.md
echo.
echo 快速开始:
echo   python starvla_act_scripts\server_starvla_act.py ^
echo     --ckpt_path output\starvla_act_g1\g1_act_task_04\final_model ^
echo     --host 0.0.0.0 ^
echo     --port 5000
pause
'''
    
    with open(package_dir / "launch.bat", "w", encoding="utf-8") as f:
        f.write(launch_bat)
    
    # 6. 创建 requirements.txt
    print("创建 requirements.txt...")
    requirements = '''torch>=2.0.0
torchvision>=0.15.0
flask>=2.3.0
pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.65.0
wandb>=0.15.0
omegaconf>=2.3.0
accelerate>=0.20.0
transformers>=4.30.0
'''
    
    with open(package_dir / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    # 7. 创建测试脚本
    print("创建测试脚本...")
    test_script = '''"""
STAR-VLA ACT 部署测试脚本
用于验证环境是否正确配置
"""
import sys
import os
from pathlib import Path

def test_environment():
    """测试环境配置"""
    print("测试环境配置...")
    
    # 1. 检查 Python 版本
    python_version = sys.version_info
    print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 版本过低，需要 3.8+")
        return False
    
    # 2. 检查依赖
    required_packages = [
        'torch', 'torchvision', 'flask', 'pillow', 'numpy', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (未安装)")
    
    if missing_packages:
        print(f"\n缺少依赖: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 3. 检查模型文件
    model_path = Path("output/starvla_act_g1/g1_act_task_04/final_model")
    if not model_path.exists():
        print(f"✗ 模型文件不存在: {model_path}")
        return False
    
    print(f"✓ 模型文件存在: {model_path}")
    
    # 4. 检查框架代码
    framework_path = Path("external/starVLA")
    if not framework_path.exists():
        print(f"✗ 框架代码不存在: {framework_path}")
        return False
    
    print(f"✓ 框架代码存在: {framework_path}")
    
    # 5. 测试导入
    try:
        sys.path.insert(0, str(framework_path))
        from deployment.model_server.policy_wrapper import PolicyServerWrapper
        print("✓ 框架导入成功")
    except ImportError as e:
        print(f"✗ 框架导入失败: {e}")
        return False
    
    print("\n✅ 环境测试通过！")
    return True

if __name__ == "__main__":
    success = test_environment()
    if not success:
        print("\n请解决上述问题后重试")
        sys.exit(1)
    else:
        print("\n可以开始使用 STAR-VLA ACT 模型了！")
'''
    
    with open(package_dir / "test_environment.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    # 8. 创建打包信息文件
    print("创建打包信息...")
    package_info = {
        "打包时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "包含内容": [
            "训练脚本: starvla_act_scripts/train_g1_act.sh",
            "推理脚本: starvla_act_scripts/inference_starvla_act.py",
            "服务器脚本: starvla_act_scripts/server_starvla_act.py",
            "模型文件: output/starvla_act_g1/*/final_model/",
            "框架代码: external/starVLA/",
            "部署文档: DEPLOYMENT_GUIDE.md",
            "启动脚本: launch.sh / launch.bat",
            "依赖列表: requirements.txt",
            "测试脚本: test_environment.py"
        ],
        "模型任务": ["g1_act_task_04", "g1_act_task_05", "g1_act_task_06"],
        "总大小MB": 0  # 稍后计算
    }
    
    # 计算总大小
    total_size = 0
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    package_info["总大小MB"] = round(total_size / (1024 * 1024), 2)
    
    with open(package_dir / "package_info.json", "w", encoding="utf-8") as f:
        json.dump(package_info, f, ensure_ascii=False, indent=2)
    
    # 9. 创建压缩包
    print("创建压缩包...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)
    
    # 显示结果
    zip_size = zip_file.stat().st_size / (1024 * 1024)
    print(f"\n打包完成！")
    print("=" * 50)
    print(f"压缩包: {zip_file.name}")
    print(f"大小: {zip_size:.2f} MB")
    print(f"位置: {zip_file.absolute()}")
    print(f"\n包含内容:")
    for item in sorted(package_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(base_dir)
            print(f"  {rel_path}")
    print(f"\n传输到新电脑后，请先阅读 DEPLOYMENT_GUIDE.md")

if __name__ == "__main__":
    create_package()
