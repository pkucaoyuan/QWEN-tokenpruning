"""
完整的 Token Pruning Pipeline 实现
扩展 QwenImageEditPipeline 以支持 Token Pruning
"""
import torch
import numpy as np
from typing import Optional, Union, List, Dict, Any, Callable, Tuple
from PIL import Image

from diffusers import QwenImageEditPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput

# 导入必要的函数
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 从本地文件导入 QwenImageEditPlusPipeline（因为可能不在安装的 diffusers 包中）
from pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

# 导入必要的函数
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning_modules import global_pruning_cache


class TokenPruningQwenImageEditPipeline(QwenImageEditPipeline):
    """
    支持 Token Pruning 的 QwenImageEditPipeline
    
    核心修改:
    1. 在去噪循环中管理 pruning 步骤
    2. 在 prepare_latents 后记录 token 长度
    3. 在每步前设置 pruning 状态
    """
    
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 1.0,  # Lightning 默认 1.0
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        增强的推理方法，支持 Token Pruning
        """
        # ===== 前期准备（与原版相同）=====
        # 定义 calculate_dimensions 函数（避免相对导入问题）
        import math
        
        def calculate_dimensions(target_area, ratio):
            width = math.sqrt(target_area * ratio)
            height = width / ratio
            width = round(width / 32) * 32
            height = round(height / 32) * 32
            return width, height, None
        
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width
        
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        
        # 检查输入
        self.check_inputs(
            prompt, height, width, negative_prompt, prompt_embeds,
            negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs, max_sequence_length
        )
        
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        
        # 定义 batch_size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device
        
        # 预处理图像
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2)
        
        # 编码 prompt
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image, prompt=prompt, prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask, device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length
        )
        
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_image, prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device, num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length
            )
        
        # ⭐ 准备 latents 并记录 token 长度
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            image, batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents
        )
        
        # ⭐ 记录 token 长度（关键信息）
        denoise_token_length = latents.shape[1]
        image_token_length = image_latents.shape[1] if image_latents is not None else 0
        total_token_length = denoise_token_length + image_token_length
        
        # 设置到 global cache
        global_pruning_cache.denoise_token_length = denoise_token_length
        global_pruning_cache.total_token_length = total_token_length
        
        print(f"\n[Token 信息]")
        print(f"  去噪 tokens: {denoise_token_length}")
        print(f"  图像 tokens: {image_token_length}")
        print(f"  总 tokens: {total_token_length}")
        
        if global_pruning_cache.enabled:
            print(f"\n[Pruning 策略]")
            print(f"  步骤 1: 完整计算（缓存）")
            print(f"  步骤 2: 使用步骤 1 缓存 ⚡")
            print(f"  步骤 3: 完整计算（缓存）")
            print(f"  步骤 4: 使用步骤 3 缓存 ⚡")
        
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size
        
        # 准备 timesteps
        # 定义辅助函数（避免相对导入问题）
        def calculate_shift(
            image_seq_len,
            base_seq_len: int = 256,
            max_seq_len: int = 4096,
            base_shift: float = 0.5,
            max_shift: float = 1.15,
        ):
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b = base_shift - m * base_seq_len
            mu = image_seq_len * m + b
            return mu
        
        # retrieve_timesteps 直接从 diffusers 导入
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
        
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        # 处理 guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        
        if self.attention_kwargs is None:
            self._attention_kwargs = {}
        
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        
        # ===== 去噪循环（增强版，支持 Token Pruning）=====
        self.scheduler.set_begin_index(0)
        
        # 重置 pruning 状态
        global_pruning_cache.current_step = 0
        global_pruning_cache.clear_caches()
        
        print(f"\n" + "=" * 70)
        print("去噪循环 (4 步)")
        print("=" * 70)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                self._current_timestep = t
                
                # ⭐ 设置当前步骤到 pruning cache
                global_pruning_cache.current_step = i
                
                # 打印步骤信息
                if global_pruning_cache.enabled:
                    if i in [0, 2]:
                        print(f"\n步骤 {i+1}/4: 完整计算 (缓存 image tokens)")
                    elif i in [1, 3]:
                        cache_from = 1 if i == 1 else 3
                        print(f"\n步骤 {i+1}/4: Token Pruning ⚡ (使用步骤 {cache_from} 的 image tokens)")
                else:
                    print(f"\n步骤 {i+1}/4: 正常计算")
                
                # 准备输入
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                # 条件预测
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, :latents.size(1)]
                
                # CFG
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, :latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
                
                # 更新 latents
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                
                # callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        # ===== 解码 =====
        self._current_timestep = None
        if output_type == "latent":
            output_image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            output_image = self.image_processor.postprocess(output_image, output_type=output_type)
        
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (output_image,)
        
        return QwenImagePipelineOutput(images=output_image)


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


class TokenPruningQwenImageEditPlusPipeline(QwenImageEditPlusPipeline):
    """
    支持 Token Pruning 的 QwenImageEditPipeline
    
    核心修改:
    1. 在去噪循环中管理 pruning 步骤
    2. 在 prepare_latents 后记录 token 长度
    3. 在每步前设置 pruning 状态
    """
    
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 1.0,  # Lightning 默认 1.0
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        print_debug_info: bool = False,
    ):
        """
        增强的推理方法，支持 Token Pruning
        """
        # ===== 前期准备（与原版相同）=====
        # 定义 calculate_dimensions 函数（避免相对导入问题）
        import math
        
        def calculate_dimensions(target_area, ratio):
            width = math.sqrt(target_area * ratio)
            height = width / ratio
            width = round(width / 32) * 32
            height = round(height / 32) * 32
            return width, height
        
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width
        
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        
        # 检查输入
        self.check_inputs(
            prompt, height, width, negative_prompt, prompt_embeds,
            negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs, max_sequence_length
        )
        
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        
        # 定义 batch_size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device
        
        # 预处理图像
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_image_sizes = []
            condition_images = []
            vae_image_sizes = []
            vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))
        
        # 编码 prompt
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        
        # ⭐ 准备 latents 并记录 token 长度
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # ⭐ 记录 token 长度（关键信息）
        denoise_token_length = latents.shape[1]
        image_token_length = image_latents.shape[1] if image_latents is not None else 0
        total_token_length = denoise_token_length + image_token_length
        
        # 设置到 global cache
        global_pruning_cache.denoise_token_length = denoise_token_length
        global_pruning_cache.total_token_length = total_token_length

        if print_debug_info:
        
            print(f"\n[Token 信息]")
            print(f"  去噪 tokens: {denoise_token_length}")
            print(f"  图像 tokens: {image_token_length}")
            print(f"  总 tokens: {total_token_length}")
            
            if global_pruning_cache.enabled:
                print(f"\n[Pruning 策略]")
                print(f"  步骤 1: 完整计算（缓存）")
                print(f"  步骤 2: 使用步骤 1 缓存 ⚡")
                print(f"  步骤 3: 完整计算（缓存）")
                print(f"  步骤 4: 使用步骤 3 缓存 ⚡")
        
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size
        
        # 准备 timesteps
        # 定义辅助函数（避免相对导入问题）
        def calculate_shift(
            image_seq_len,
            base_seq_len: int = 256,
            max_seq_len: int = 4096,
            base_shift: float = 0.5,
            max_shift: float = 1.15,
        ):
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            b = base_shift - m * base_seq_len
            mu = image_seq_len * m + b
            return mu
        
        # retrieve_timesteps 直接从 diffusers 导入
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
        
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        # 处理 guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        
        if self.attention_kwargs is None:
            self._attention_kwargs = {}
        
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        
        # ===== 去噪循环（增强版，支持 Token Pruning）=====
        self.scheduler.set_begin_index(0)
        
        # 重置 pruning 状态
        global_pruning_cache.current_step = 0
        global_pruning_cache.clear_caches()
        
        if print_debug_info:
            print(f"\n" + "=" * 70)
            print("去噪循环 (4 步)")
            print("=" * 70)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                self._current_timestep = t
                
                # ⭐ 设置当前步骤到 pruning cache
                global_pruning_cache.current_step = i
                
                # 打印步骤信息
                if global_pruning_cache.enabled:
                    if i in [0, 2]:
                        if print_debug_info:
                            print(f"\n步骤 {i+1}/4: 完整计算 (缓存 image tokens)")
                    elif i in [1, 3]:
                        cache_from = 1 if i == 1 else 3
                        if print_debug_info:
                            print(f"\n步骤 {i+1}/4: Token Pruning ⚡ (使用步骤 {cache_from} 的 image tokens)")
                else:
                    if print_debug_info:
                        print(f"\n步骤 {i+1}/4: 正常计算")
                
                # 准备输入
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                # 条件预测
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, :latents.size(1)]
                
                # CFG
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, :latents.size(1)]
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
                
                # 更新 latents
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                
                # callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        # ===== 解码 =====
        self._current_timestep = None
        if output_type == "latent":
            output_image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            output_image = self.image_processor.postprocess(output_image, output_type=output_type)
        
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (output_image,)
        
        return QwenImagePipelineOutput(images=output_image)