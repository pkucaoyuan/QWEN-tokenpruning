"""
步骤4: 使用融合后的 Lightning 模型（快速推理）
前提: 已运行 3_merge_lora_to_weights.py 融合 LoRA
优势: 无需每次加载 LoRA，加载速度更快
"""
import torch
import os
import argparse
from datetime import datetime
from PIL import Image
from diffusers import QwenImageEditPipeline

def load_merged_pipeline(model_path="./models/qwen-image-edit-lightning-merged"):
    """
    加载融合后的 Lightning 模型
    """
    print("=" * 60)
    print("加载融合后的 Lightning 模型")
    print("=" * 60)
    
    print(f"\n模型路径: {model_path}")
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 模型路径不存在!")
        print(f"   请先运行: python 3_merge_lora_to_weights.py")
        print(f"   来生成融合后的模型")
        return None
    
    print("正在加载模型...")
    # ⭐ 使用 low_cpu_mem_usage 加载大模型
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # 低内存模式
        variant="bf16"  # 指定精度变体
    )
    
    print("移动到 CUDA...")
    pipe.to("cuda")
    
    print("\n✅ 融合模型加载完成（无需加载 LoRA）")
    print("   加载速度更快，推理性能相同")
    
    return pipe

def run_inference(
    pipe,
    image_path="input.png",
    prompt="Change the rabbit's color to purple, with a flash light background.",
    output_dir="outputs",
    num_steps=4,
    cfg_scale=1.0
):
    """
    运行推理
    """
    print("\n" + "=" * 60)
    print("运行推理")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_merged_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # 加载输入图像
    print(f"\n1. 输入图像: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"   尺寸: {image.size}")
    except Exception as e:
        print(f"   ❌ 错误: 无法加载图像 - {e}")
        return None, None
    
    # 准备推理参数
    print(f"\n2. 推理参数:")
    print(f"   - Prompt: {prompt}")
    print(f"   - 步数: {num_steps}")
    print(f"   - CFG Scale: {cfg_scale}")
    
    inference_params = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": " ",
        "num_inference_steps": num_steps,
        "true_cfg_scale": cfg_scale,
        "generator": torch.manual_seed(0),
    }
    
    # 执行推理
    print(f"\n3. 执行推理...")
    with torch.inference_mode():
        output = pipe(**inference_params)
        output_image = output.images[0]
    
    # 保存结果
    print(f"\n4. 保存结果...")
    output_image.save(output_path)
    print(f"   文件: {output_path}")
    print(f"   时间戳: {timestamp}")
    
    # 同时保存最新版本
    latest_path = os.path.join(output_dir, "latest_output_merged.png")
    output_image.save(latest_path)
    print(f"   最新: {latest_path}")
    
    print("\n✅ 推理完成！")
    return output_image, output_path

def main():
    """
    主流程
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='使用融合后的 Qwen-Image-Edit Lightning 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础用法
  python 4_use_merged_model.py
  
  # 自定义输入和 prompt
  python 4_use_merged_model.py -i my_photo.jpg -p "Make the background blue"
  
  # 完整参数
  python 4_use_merged_model.py -i image.png -p "Add sunglasses" -o results -s 4 -c 1.0
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
    parser.add_argument('--model_path', '-m', type=str,
                        default='./models/qwen-image-edit-lightning-merged',
                        help='融合模型路径')
    
    args = parser.parse_args()
    
    # 加载融合后的模型
    pipe = load_merged_pipeline(args.model_path)
    
    if pipe is None:
        print("\n❌ 模型加载失败，退出")
        return
    
    # 运行推理
    output_image, output_path = run_inference(
        pipe,
        image_path=args.input,
        prompt=args.prompt,
        output_dir=args.output_dir,
        num_steps=args.steps,
        cfg_scale=args.cfg
    )
    
    if output_path:
        print("\n" + "=" * 60)
        print("✅ 完成！")
        print("=" * 60)
        print(f"\n输出文件: {output_path}")
        print(f"查看输出: ls -l {output_path}")

if __name__ == "__main__":
    main()

