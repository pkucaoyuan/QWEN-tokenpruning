"""
完整的 Token Pruning 推理脚本
实现策略: 步骤 1,3 完整计算; 步骤 2,4 使用缓存

使用方法:
  python run_with_token_pruning.py -i input.png -p "Your prompt"
  python run_with_token_pruning.py -i input.png -p "Your prompt" --no-pruning  # 对比基线
"""
import sys
import os

# 添加当前目录到 path，以便导入 pruning_modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import math
import argparse
import time
from datetime import datetime
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

# 导入 pruning 模块和自定义 pipeline
from pruning_modules import (
    global_pruning_cache,
    apply_token_pruning_to_transformer
)
from pruning_pipeline_full import TokenPruningQwenImageEditPipeline


def setup_pipeline_with_pruning(enable_pruning=True):
    """
    设置带 Token Pruning 的 Pipeline
    """
    print("=" * 70)
    print("设置 Qwen-Image-Edit Lightning Pipeline")
    if enable_pruning:
        print("Token Pruning: ✅ 启用 (步骤 1,3 完整; 步骤 2,4 缓存)")
    else:
        print("Token Pruning: ❌ 禁用 (基线对比)")
    print("=" * 70)
    
    # 1. 配置调度器
    print("\n[1/5] 配置 FlowMatchEulerDiscreteScheduler...")
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # 2. 加载基础模型（使用自定义 Pipeline）
    print("\n[2/5] 加载基础模型: Qwen/Qwen-Image-Edit...")
    pipe = TokenPruningQwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )
    
    # 3. 加载 Lightning LoRA
    print("\n[3/5] 加载 Lightning LoRA 权重...")
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
    )
    print("   ✅ LoRA 加载成功")
    
    # 4. 应用 Token Pruning（如果启用）
    if enable_pruning:
        print("\n[4/5] 应用 Token Pruning 到 Transformer...")
        apply_token_pruning_to_transformer(pipe.transformer)
        global_pruning_cache.enabled = True
    else:
        print("\n[4/5] 跳过 Token Pruning（基线模式）")
        global_pruning_cache.enabled = False
    
    # 5. 移动到 CUDA
    print("\n[5/5] 移动到 CUDA...")
    pipe.to("cuda")
    
    print("\n" + "=" * 70)
    print("✅ Pipeline 设置完成！")
    print("=" * 70)
    
    return pipe


def run_inference_with_pruning(
    pipe,
    image_path,
    prompt,
    output_dir="outputs_pruning",
    num_steps=4,
    cfg_scale=1.0,
    enable_pruning=True
):
    """
    运行推理（带 Token Pruning）
    """
    print("\n" + "=" * 70)
    print("开始推理")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    print(f"\n[输入] 图像: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"       尺寸: {image.size}")
    except Exception as e:
        print(f"❌ 错误: 无法加载图像 - {e}")
        return None, None, None
    
    print(f"[输入] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"[参数] 推理步数: {num_steps}")
    print(f"[参数] CFG Scale: {cfg_scale}")
    print(f"[参数] Token Pruning: {'启用' if enable_pruning else '禁用 (基线对比)'}")
    
    # 准备推理参数
    inference_params = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": " ",
        "num_inference_steps": num_steps,
        "true_cfg_scale": cfg_scale,
        "generator": torch.manual_seed(42),
    }
    
    # 重置 pruning 状态
    global_pruning_cache.clear_caches()
    global_pruning_cache.current_step = 0
    
    # 执行推理
    print("\n" + "-" * 70)
    print(f"{'推理过程 (Token Pruning)' if enable_pruning else '推理过程 (Baseline)'}:")
    print("-" * 70)
    
    # ⏱️ 开始计时
    print("\n⏱️  计时开始...")
    inference_start = time.time()
    
    try:
        # 使用自定义 Pipeline 的 __call__ 方法
        # Token 长度信息会在内部自动设置
        with torch.inference_mode():
            output = pipe(**inference_params)
            output_image = output.images[0]
    
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # ⏱️ 结束计时
    inference_time = time.time() - inference_start
    print(f"\n⏱️  推理完成，耗时: {inference_time:.2f} 秒")
    
    # 🔬 打印缓存操作的详细统计
    if enable_pruning:
        global_pruning_cache.print_timing_stats()
    
    # 保存结果
    print("\n" + "-" * 70)
    print("保存结果:")
    print("-" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "pruning" if enable_pruning else "baseline"
    output_filename = f"output_{suffix}_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    output_image.save(output_path)
    print(f"✅ 文件: {output_path}")
    
    # 保存最新版本
    latest_path = os.path.join(output_dir, f"latest_{suffix}.png")
    output_image.save(latest_path)
    print(f"   最新: {latest_path}")
    
    # 时间统计
    print(f"\n" + "=" * 70)
    print(f"⏱️  性能统计:")
    print("=" * 70)
    print(f"  推理时间: {inference_time:.2f} 秒")
    print(f"  模式: {'Token Pruning' if enable_pruning else 'Baseline (无优化)'}")
    
    return output_image, output_path, inference_time


def main():
    """
    主程序
    """
    parser = argparse.ArgumentParser(
        description='Qwen-Image-Edit Lightning + Token Pruning 完整实现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启用 Token Pruning
  python run_with_token_pruning.py -i input.png -p "Make it purple"
  
  # 禁用 Pruning（对比基线）
  python run_with_token_pruning.py -i input.png -p "Make it purple" --no-pruning
  
  # 对比实验
  python run_with_token_pruning.py -p "Your prompt" --no-pruning  # 运行基线
  python run_with_token_pruning.py -p "Your prompt"              # 运行 pruning
  # 对比 outputs_pruning/ 中的两个输出
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default='input.png',
                        help='输入图片路径 (默认: input.png)')
    parser.add_argument('--prompt', '-p', type=str,
                        default='Change the rabbit\'s color to purple',
                        help='编辑指令')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs_pruning',
                        help='输出目录 (默认: outputs_pruning)')
    parser.add_argument('--steps', '-s', type=int, default=4,
                        help='推理步数 (默认: 4)')
    parser.add_argument('--cfg', '-c', type=float, default=1.0,
                        help='CFG Scale (默认: 1.0)')
    parser.add_argument('--no-pruning', action='store_true',
                        help='禁用 Token Pruning（用于对比基线）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    # 设置 Pipeline
    pipe = setup_pipeline_with_pruning(enable_pruning=not args.no_pruning)
    
    # 运行推理
    output_image, output_path, inference_time = run_inference_with_pruning(
        pipe,
        image_path=args.input,
        prompt=args.prompt,
        output_dir=args.output_dir,
        num_steps=args.steps,
        cfg_scale=args.cfg,
        enable_pruning=not args.no_pruning
    )
    
    if output_path:
        print("\n" + "=" * 70)
        print("✅ 实验完成！")
        print("=" * 70)
        
        mode_name = "Token Pruning" if not args.no_pruning else "Baseline"
        print(f"\n📊 实验结果:")
        print(f"  模式: {mode_name}")
        print(f"  推理时间: {inference_time:.2f} 秒")
        print(f"  输出文件: {output_path}")
        
        if not args.no_pruning:
            print("\n💡 提示: 运行基线对比以评估加速效果:")
            print(f"  python run_with_token_pruning.py \\")
            print(f"      -i {args.input} \\")
            print(f"      -p \"{args.prompt[:50]}...\" \\")
            print(f"      --no-pruning")
            print(f"\n  然后对比:")
            print(f"    outputs_pruning/latest_pruning.png  ← Token Pruning")
            print(f"    outputs_pruning/latest_baseline.png ← Baseline")
        else:
            print("\n💡 提示: 运行 Token Pruning 版本:")
            print(f"  python run_with_token_pruning.py \\")
            print(f"      -i {args.input} \\")
            print(f"      -p \"{args.prompt[:50]}...\"")
            print(f"\n  查看加速效果和质量对比")


if __name__ == "__main__":
    main()

