"""
测试自定义 Processor 的真实开销
通过对比原始 Pipeline 和自定义 Processor（不启用 pruning）的性能
"""
import torch
import time
from diffusers import DiffusionPipeline
from PIL import Image
import sys

def test_baseline():
    """测试原始 Baseline（不使用任何自定义代码）"""
    print("=" * 70)
    print("测试 1: Baseline Pipeline（原始实现）")
    print("=" * 70)
    
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    
    # 加载 Lightning LoRA
    pipe.load_lora_weights("./models", weight_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
    
    # 加载图像
    input_image = Image.open("input.png").convert("RGB")
    
    # 预热
    print("\n预热...")
    _ = pipe(
        prompt="test",
        image=input_image,
        height=1080,
        width=1620,
        num_inference_steps=4,
        guidance_scale=1.0,
    ).images[0]
    
    # 正式测试（3次取平均）
    print("\n正式测试（3次）...")
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        start = time.time()
        
        _ = pipe(
            prompt="Convert the male person to female",
            image=input_image,
            height=1080,
            width=1620,
            num_inference_steps=4,
            guidance_scale=1.0,
        ).images[0]
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  运行 {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n✅ Baseline 平均时间: {avg_time:.2f}s")
    
    return avg_time


def test_custom_processor_without_pruning():
    """测试自定义 Processor（不启用 pruning）"""
    print("\n" + "=" * 70)
    print("测试 2: 自定义 Processor（Pruning 禁用）")
    print("=" * 70)
    
    # 导入自定义 pipeline
    sys.path.insert(0, '.')
    from pruning_pipeline_full import TokenPruningQwenImageEditPipeline
    from pruning_modules import global_pruning_cache
    
    pipe = TokenPruningQwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    
    # 加载 Lightning LoRA
    pipe.load_lora_weights("./models", weight_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
    
    # ⚠️ 关键：不启用 pruning
    print("\n⚠️  Pruning: 禁用（测试纯 Processor 开销）")
    
    # 加载图像
    input_image = Image.open("input.png").convert("RGB")
    
    # 预热
    print("\n预热...")
    _ = pipe(
        prompt="test",
        image=input_image,
        height=1080,
        width=1620,
        num_inference_steps=4,
        guidance_scale=1.0,
        enable_pruning=False,  # 禁用 pruning
    ).images[0]
    
    # 正式测试（3次取平均）
    print("\n正式测试（3次）...")
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        start = time.time()
        
        _ = pipe(
            prompt="Convert the male person to female",
            image=input_image,
            height=1080,
            width=1620,
            num_inference_steps=4,
            guidance_scale=1.0,
            enable_pruning=False,  # 禁用 pruning
        ).images[0]
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  运行 {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n✅ 自定义 Processor（无 Pruning）平均时间: {avg_time:.2f}s")
    
    return avg_time


def test_custom_processor_with_pruning():
    """测试自定义 Processor（启用 pruning）"""
    print("\n" + "=" * 70)
    print("测试 3: 自定义 Processor（Pruning 启用）")
    print("=" * 70)
    
    # 导入自定义 pipeline
    sys.path.insert(0, '.')
    from pruning_pipeline_full import TokenPruningQwenImageEditPipeline
    from pruning_modules import global_pruning_cache
    
    pipe = TokenPruningQwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    
    # 加载 Lightning LoRA
    pipe.load_lora_weights("./models", weight_name="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
    
    print("\n✅ Pruning: 启用")
    
    # 加载图像
    input_image = Image.open("input.png").convert("RGB")
    
    # 预热
    print("\n预热...")
    _ = pipe(
        prompt="test",
        image=input_image,
        height=1080,
        width=1620,
        num_inference_steps=4,
        guidance_scale=1.0,
        enable_pruning=True,  # 启用 pruning
    ).images[0]
    
    # 正式测试（3次取平均）
    print("\n正式测试（3次）...")
    times = []
    for i in range(3):
        torch.cuda.synchronize()
        start = time.time()
        
        _ = pipe(
            prompt="Convert the male person to female",
            image=input_image,
            height=1080,
            width=1620,
            num_inference_steps=4,
            guidance_scale=1.0,
            enable_pruning=True,  # 启用 pruning
        ).images[0]
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  运行 {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n✅ 自定义 Processor（启用 Pruning）平均时间: {avg_time:.2f}s")
    
    return avg_time


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔬 精确测量自定义 Processor 的开销")
    print("=" * 70)
    
    # 测试 1: Baseline
    baseline_time = test_baseline()
    
    # 测试 2: 自定义 Processor（不启用 pruning）
    custom_no_pruning_time = test_custom_processor_without_pruning()
    
    # 测试 3: 自定义 Processor（启用 pruning）
    custom_with_pruning_time = test_custom_processor_with_pruning()
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("📊 结果汇总")
    print("=" * 70)
    print(f"1. Baseline:                    {baseline_time:.2f}s")
    print(f"2. 自定义 Processor（无 Pruning）: {custom_no_pruning_time:.2f}s")
    print(f"3. 自定义 Processor（启用 Pruning）: {custom_with_pruning_time:.2f}s")
    print()
    print(f"自定义 Processor 本身的开销:    {custom_no_pruning_time - baseline_time:+.2f}s ({(custom_no_pruning_time/baseline_time-1)*100:+.1f}%)")
    print(f"Pruning 的净效果:              {custom_with_pruning_time - custom_no_pruning_time:+.2f}s ({(custom_with_pruning_time/custom_no_pruning_time-1)*100:+.1f}%)")
    print(f"总体效果（vs Baseline）:        {custom_with_pruning_time - baseline_time:+.2f}s ({(custom_with_pruning_time/baseline_time-1)*100:+.1f}%)")
    print("=" * 70)

