"""
步骤0: 环境检查和自动修复
运行此脚本检查并修复环境问题
"""
import subprocess
import sys

def check_and_fix_environment():
    """
    检查并修复环境依赖
    """
    print("=" * 60)
    print("Qwen-Image-Edit 环境检查和修复")
    print("=" * 60)
    
    # 1. 检查 Python 版本
    print("\n1. 检查 Python 版本...")
    print(f"   Python 版本: {sys.version}")
    if sys.version_info < (3, 8):
        print("   ❌ Python 版本过低，需要 3.8+")
        return
    print("   ✅ Python 版本符合要求")
    
    # 2. 检查并安装 diffusers
    print("\n2. 检查 diffusers...")
    try:
        import diffusers
        print(f"   当前版本: {diffusers.__version__}")
        if "0.36" not in diffusers.__version__:
            print("   ⚠️ 建议升级到最新版本")
            print("   运行: pip install git+https://github.com/huggingface/diffusers")
        else:
            print("   ✅ diffusers 版本正确")
    except ImportError:
        print("   ❌ diffusers 未安装")
        print("   正在安装...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/huggingface/diffusers"
        ])
    
    # 3. 检查并修复 transformers
    print("\n3. 检查 transformers...")
    try:
        import transformers
        version = transformers.__version__
        print(f"   当前版本: {version}")
        
        # 检查版本是否足够新
        major, minor = map(int, version.split('.')[:2])
        if major < 4 or (major == 4 and minor < 48):
            print("   ⚠️ transformers 版本过低，需要升级")
            print("   正在升级 transformers...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--upgrade", "transformers>=4.48.0"
            ])
            print("   ✅ transformers 已升级")
        else:
            print("   ✅ transformers 版本符合要求")
    except ImportError:
        print("   ❌ transformers 未安装")
        print("   正在安装...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "transformers>=4.48.0"
        ])
    
    # 4. 检查 torch
    print("\n4. 检查 PyTorch...")
    try:
        import torch
        print(f"   PyTorch 版本: {torch.__version__}")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print("   ✅ PyTorch 配置正确")
        else:
            print("   ⚠️ CUDA 不可用，将使用 CPU（速度很慢）")
    except ImportError:
        print("   ❌ PyTorch 未安装")
        print("   请手动安装: pip install torch")
    
    # 5. 检查其他依赖
    print("\n5. 检查其他依赖...")
    dependencies = {
        "PIL": "pillow",
        "accelerate": "accelerate",
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface_hub",
        "peft": "peft",  # ⭐ LoRA 加载必需
    }
    
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} 未安装，正在安装...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name
            ])
    
    # 6. 检查输入图片
    print("\n6. 检查输入图片...")
    import os
    if os.path.exists("input.png"):
        print("   ✅ input.png 存在")
    elif os.path.exists("input.jpg"):
        print("   ⚠️ 发现 input.jpg，正在转换为 input.png...")
        from PIL import Image
        Image.open("input.jpg").convert("RGB").save("input.png")
        print("   ✅ 转换完成")
    else:
        print("   ⚠️ 未找到输入图片")
        print("   请准备 input.png 或 input.jpg")
    
    print("\n" + "=" * 60)
    print("✅ 环境检查完成！")
    print("=" * 60)
    print("\n可以运行: python 2_load_and_inference_lightning.py")

if __name__ == "__main__":
    check_and_fix_environment()

