"""
步骤3: 将 LoRA 权重融合到基础模型（可选）
融合后可以直接使用，无需每次加载 LoRA
"""
import torch
import math
from diffusers import QwenImageEditPipeline, FlowMatchEulerDiscreteScheduler

def merge_lora_and_save():
    """
    融合 LoRA 权重到基础模型并保存
    """
    print("=" * 60)
    print("融合 Lightning LoRA 到基础模型")
    print("=" * 60)
    
    # 步骤1: 配置调度器
    print("\n1. 配置调度器...")
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
    
    # 步骤2: 加载基础模型
    print("\n2. 加载基础模型...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )
    
    # 步骤3: 加载 LoRA 权重
    print("\n3. 加载 Lightning LoRA 权重...")
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
    )
    
    # 步骤4: 融合 LoRA（⭐ 关键步骤）
    print("\n4. 融合 LoRA 权重到基础模型...")
    pipe.fuse_lora(lora_scale=1.0)
    print("   ✅ LoRA 已融合")
    
    # 移动到 CPU 以便正确保存
    print("\n4.5 准备保存（移动到 CPU）...")
    pipe.to("cpu")
    
    # 步骤5: 保存融合后的模型
    output_dir = "./models/qwen-image-edit-lightning-merged"
    print(f"\n5. 保存融合后的模型到: {output_dir}")
    pipe.save_pretrained(output_dir, safe_serialization=True)
    
    print("\n✅ 完成！融合后的模型已保存")
    print(f"\n之后可以直接加载融合后的模型：")
    print(f"pipe = QwenImageEditPipeline.from_pretrained('{output_dir}')")
    
    return output_dir

def test_merged_model(model_dir):
    """
    测试融合后的模型
    """
    print("\n" + "=" * 60)
    print("测试融合后的模型")
    print("=" * 60)
    
    print(f"\n加载融合后的模型: {model_dir}")
    pipe = QwenImageEditPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    print("\n✅ 融合后的模型加载成功！")
    print("可以直接使用，无需再加载 LoRA")
    
    return pipe

if __name__ == "__main__":
    # 融合并保存
    merged_dir = merge_lora_and_save()
    
    # 测试加载
    # test_merged_model(merged_dir)

