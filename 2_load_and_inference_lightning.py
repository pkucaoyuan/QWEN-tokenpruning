"""
步骤2: 加载 Lightning LoRA 并运行推理
按照官方 Hugging Face 指引实现
"""
import torch
import math
import os
import argparse
from datetime import datetime
from PIL import Image
from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler

def setup_lightning_pipeline():
    """
    按照官方指引设置 Lightning Pipeline
    """
    print("=" * 60)
    print("设置 Qwen-Image-Edit Lightning Pipeline")
    print("=" * 60)
    
    # 步骤1: 配置调度器（按照官方配置）
    print("\n1. 配置 FlowMatchEulerDiscreteScheduler...")
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # 官方推荐配置
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,  # 重要：启用动态 shifting
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # 步骤2: 加载基础 Qwen-Image-Edit 模型
    print("\n2. 加载基础模型: Qwen/Qwen-Image-Edit...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        scheduler=scheduler,  # 使用配置的调度器
        torch_dtype=torch.bfloat16
    )
    
    # 步骤3: 加载 Lightning LoRA 权重
    print("\n3. 加载 Lightning LoRA 权重...")
    try:
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
        )
        print("   ✅ LoRA 权重加载成功")
    except Exception as e:
        print(f"   ❌ LoRA 加载失败: {e}")
        print("   尝试从本地加载...")
        pipe.load_lora_weights(
            "./models/lightning_lora/Qwen-Image-Edit-2509",
            weight_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
        )
    
    # 步骤4: 移动到 GPU
    print("\n4. 移动到 CUDA...")
    pipe.to("cuda")
    
    print("\n✅ Pipeline 设置完成！")
    return pipe

def run_lightning_inference(
    pipe,
    image_path="input.png",
    prompt="Change the rabbit's color to purple, with a flash light background.",
    output_dir="outputs",
    num_steps=4,
    cfg_scale=1.0
):
    """
    使用 Lightning 运行推理
    """
    print("\n" + "=" * 60)
    print("运行 Lightning 推理")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_lightning_4steps_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # 加载输入图像
    print(f"\n1. 加载输入图像: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"   图像尺寸: {image.size}")
    
    # 准备推理参数（按照官方示例）
    print("\n2. 准备推理参数...")
    inference_params = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": " ",  # 空字符串
        "num_inference_steps": num_steps,  # ⭐ 从参数获取
        "true_cfg_scale": cfg_scale,  # ⭐ 从参数获取
        "generator": torch.manual_seed(0),
    }
    
    print(f"   - Prompt: {prompt}")
    print(f"   - 推理步数: {num_steps}")
    print(f"   - CFG Scale: {cfg_scale}")
    
    # 执行推理
    print("\n3. 执行推理...")
    with torch.inference_mode():
        output = pipe(**inference_params)
        output_image = output.images[0]
    
    # 保存结果
    print(f"\n4. 保存结果...")
    output_image.save(output_path)
    print(f"   文件: {output_path}")
    print(f"   时间戳: {timestamp}")
    
    # 同时保存一个不带时间戳的版本（方便查看最新）
    latest_path = os.path.join(output_dir, "latest_output.png")
    output_image.save(latest_path)
    print(f"   最新: {latest_path}")
    
    print("\n✅ 推理完成！")
    return output_image, output_path

def main():
    """
    主流程
    """
    # ⭐ 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Qwen-Image-Edit Lightning 推理（4步快速）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础用法
  python 2_load_and_inference_lightning.py
  
  # 自定义输入和 prompt
  python 2_load_and_inference_lightning.py -i my_photo.jpg -p "Make it purple"
  
  # 完整参数
  python 2_load_and_inference_lightning.py -i image.png -p "Add rainbow" -o results -s 4 -c 1.0
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default='input.png',
                        help='输入图片路径 (默认: input.png)')
    parser.add_argument('--prompt', '-p', type=str,
                        default='Change the rabbit\'s color to purple, with a flash light background.',
                        help='编辑指令')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--steps', '-s', type=int, default=4,
                        help='推理步数 (默认: 4, Lightning 推荐)')
    parser.add_argument('--cfg', '-c', type=float, default=1.0,
                        help='CFG Scale (默认: 1.0, Lightning 推荐)')
    
    args = parser.parse_args()
    
    # 设置 Pipeline
    pipe = setup_lightning_pipeline()
    
    # 运行推理（使用命令行参数）
    output_image, output_path = run_lightning_inference(
        pipe,
        image_path=args.input,     # ⭐ 从命令行获取
        prompt=args.prompt,         # ⭐ 从命令行获取
        output_dir=args.output_dir  # ⭐ 从命令行获取
    )
    
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"\n输入: {args.input}")
    print(f"Prompt: {args.prompt}")
    print(f"输出: {output_path}")
    print(f"\n查看输出: ls -l {output_path}")

if __name__ == "__main__":
    main()

