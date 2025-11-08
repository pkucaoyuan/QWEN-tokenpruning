"""
Token Pruning 核心模块
实现完整的 Token Pruning 功能，包括修改的 Transformer Block 和 Attention Processor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn


class TokenPruningCache:
    """
    管理 Token Pruning 的缓存
    
    缓存策略:
    - 步骤 0 (第1步): 完整计算，缓存所有层的 image tokens hidden states
    - 步骤 1 (第2步): 使用步骤 0 的缓存
    - 步骤 2 (第3步): 完整计算，缓存所有层的 image tokens hidden states
    - 步骤 3 (第4步): 使用步骤 2 的缓存
    """
    
    def __init__(self):
        self.enabled = False
        self.current_step = 0  # 当前去噪步骤 (0-3)
        self.denoise_token_length = None  # 去噪 tokens 数量
        self.total_token_length = None    # 总 tokens 数量
        
        # 每层的缓存: {layer_idx: {step: hidden_states}}
        # 例如: layer_caches[0][0] = 第0层在步骤0的 image tokens hidden states
        self.layer_caches = {}  # {layer_idx: {0: tensor, 2: tensor}}
        
    def should_prune_current_step(self) -> bool:
        """判断当前步骤是否需要 prune"""
        if not self.enabled:
            return False
        # 步骤 1 和 3 (索引 1, 3) 需要使用缓存
        return self.current_step in [1, 3]
    
    def get_cache_step_idx(self) -> Optional[int]:
        """获取应该使用哪一步的缓存"""
        if self.current_step == 1:
            return 0  # 步骤 2 使用步骤 1 的缓存
        elif self.current_step == 3:
            return 2  # 步骤 4 使用步骤 3 的缓存
        return None
    
    def should_cache_current_step(self) -> bool:
        """判断当前步骤是否需要缓存"""
        if not self.enabled:
            return False
        # 步骤 0 和 2 (步骤 1 和 3) 需要缓存
        return self.current_step in [0, 2]
    
    def _init_buffers_if_needed(self, layer_idx: int, image_k: torch.Tensor, image_v: torch.Tensor, image_hidden: torch.Tensor = None):
        """⚡ 预分配缓存 buffer（只在第一次调用时）"""
        if layer_idx in self._preallocated_buffers:
            return
        
        # 为每一层预分配 2 个步骤的缓存（步骤 0 和 2）
        self._preallocated_buffers[layer_idx] = {
            0: {
                'k': torch.empty_like(image_k),
                'v': torch.empty_like(image_v),
                'hidden': torch.empty_like(image_hidden) if image_hidden is not None else None,
            },
            2: {
                'k': torch.empty_like(image_k),
                'v': torch.empty_like(image_v),
                'hidden': torch.empty_like(image_hidden) if image_hidden is not None else None,
            }
        }
    
    def cache_layer_kv(self, layer_idx: int, image_k: torch.Tensor, image_v: torch.Tensor, image_hidden: torch.Tensor = None):
        """⚡ 缓存某一层的 image tokens K, V（使用预分配的 buffer，避免 cudaMalloc）"""
        if not self.should_cache_current_step():
            return
        
        # ⚡ 第一次调用时初始化 buffer
        self._init_buffers_if_needed(layer_idx, image_k, image_v, image_hidden)
        
        if layer_idx not in self.layer_caches:
            self.layer_caches[layer_idx] = {}
        
        # ⚡ 使用预分配的 buffer，通过 copy_() 进行 in-place 拷贝
        # 避免 clone() 触发新的内存分配
        buffer = self._preallocated_buffers[layer_idx][self.current_step]
        buffer['k'].copy_(image_k)
        buffer['v'].copy_(image_v)
        if image_hidden is not None and buffer['hidden'] is not None:
            buffer['hidden'].copy_(image_hidden)
        
        # 存储 buffer 的引用
        self.layer_caches[layer_idx][self.current_step] = buffer
    
    def update_layer_hidden(self, layer_idx: int, image_hidden: torch.Tensor):
        """⚡ 更新某一层缓存中的 hidden states（使用预分配的 buffer）"""
        if not self.should_cache_current_step():
            return
        
        if layer_idx in self.layer_caches and self.current_step in self.layer_caches[layer_idx]:
            buffer = self.layer_caches[layer_idx][self.current_step]
            
            # ⚡ 如果 buffer 的 hidden 还没初始化（第一次遇到 hidden）
            if buffer['hidden'] is None:
                buffer['hidden'] = torch.empty_like(image_hidden)
            
            # ⚡ 使用 copy_() 进行 in-place 拷贝
            buffer['hidden'].copy_(image_hidden)
    
    def get_cached_layer_kv(self, layer_idx: int):
        """获取某一层的缓存 image tokens K, V 和 hidden states（全部在 GPU 上）"""
        if not self.should_prune_current_step():
            return None, None, None
        
        cache_step = self.get_cache_step_idx()
        if cache_step is None:
            return None, None, None
        
        if layer_idx not in self.layer_caches:
            return None, None, None
        
        if cache_step not in self.layer_caches[layer_idx]:
            return None, None, None
        
        cache_dict = self.layer_caches[layer_idx][cache_step]
        # ⭐ 确保返回的缓存在 GPU 上
        return cache_dict['k'], cache_dict['v'], cache_dict['hidden']
    
    def clear_caches(self):
        """清空所有缓存（但保留预分配的 buffer）"""
        self.layer_caches = {}
        # ⚡ 注意：不清空 _preallocated_buffers，复用已分配的内存
    
    def get_image_token_slice(self):
        """获取 image tokens 的切片范围"""
        if self.denoise_token_length is None:
            return None
        return slice(self.denoise_token_length, self.total_token_length)


# 全局 pruning cache 实例
global_pruning_cache = TokenPruningCache()


# ⚡ 全局 RoPE 函数（避免在每层重复定义）
def apply_rotary_emb_qwen(x, freqs, use_real=True):
    """简化的 RoPE 实现"""
    if use_real:
        cos, sin = freqs
        cos = cos[None, None].to(x.device)
        sin = sin[None, None].to(x.device)
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = freqs.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3)
        return x_out.type_as(x)


class PrunableQwenDoubleStreamAttnProcessor:
    """
    支持 Token Pruning 的双流注意力处理器
    
    Pruning 策略:
    - Image tokens 不计算 Q（不需要主动查询）
    - Image tokens 使用缓存的 hidden states 计算 K, V（提供被查询的信息）
    - 去噪 tokens 正常计算 Q, K, V
    """
    
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("需要 PyTorch 2.0+")
        self._attention_backend = None
        self._parallel_config = None
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        带 Token Pruning 的注意力计算
        """
        if encoder_hidden_states is None:
            raise ValueError("需要 encoder_hidden_states (text stream)")
        
        # 检查是否需要 pruning
        should_prune = global_pruning_cache.should_prune_current_step()
        should_cache = global_pruning_cache.should_cache_current_step()
        L_denoise = global_pruning_cache.denoise_token_length
        
        seq_txt = encoder_hidden_states.shape[1]
        
        # ===== 获取缓存的 K, V（如果是 pruning 步骤）=====
        cached_image_k, cached_image_v, cached_image_hidden = None, None, None
        if should_prune and L_denoise is not None:
            # 尝试从缓存获取
            layer_idx = getattr(attn, '_layer_idx', None)
            if layer_idx is not None:
                cached_image_k, cached_image_v, cached_image_hidden = global_pruning_cache.get_cached_layer_kv(layer_idx)
        
        # ===== 计算 QKV =====
        if should_prune and cached_image_k is not None and cached_image_v is not None:
            # ⚡ Pruning 模式：使用缓存的 K, V，不重新计算！
            denoise_hidden = hidden_states[:, :L_denoise]
            
            # 去噪 tokens: 完整 QKV
            denoise_query = attn.to_q(denoise_hidden)
            denoise_key = attn.to_k(denoise_hidden)
            denoise_value = attn.to_v(denoise_hidden)
            
            # 图像 tokens: ⚡⚡⚡ 直接使用缓存，完全跳过计算！
            # 不需要 image_query（去噪 tokens 不 attend 到 image tokens）
            
            # 合并 Q, K, V
            img_query = denoise_query  # 只有去噪部分有 Q
            img_key = torch.cat([denoise_key, cached_image_k], dim=1)     # ⚡ 使用缓存
            img_value = torch.cat([denoise_value, cached_image_v], dim=1)  # ⚡ 使用缓存
            
        else:
            # 正常模式：完整计算
            img_query = attn.to_q(hidden_states)
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)
            
            # ⚡ 如果需要缓存，保存 image tokens 的 K, V
            if should_cache and L_denoise is not None:
                layer_idx = getattr(attn, '_layer_idx', None)
                if layer_idx is not None:
                    # ⚡ 直接传递 slice（view），cache_layer_kv 会使用 copy_() 到预分配的 buffer
                    # 避免 clone() 触发新的内存分配
                    image_k = img_key[:, L_denoise:]
                    image_v = img_value[:, L_denoise:]
                    # 暂时用 None 占位，Block 层面会更新
                    global_pruning_cache.cache_layer_kv(layer_idx, image_k, image_v, None)
        
        # 文本流：始终正常计算
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)
        
        # ===== Reshape for multi-head attention =====
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))
        
        # ===== QK normalization =====
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)
        
        # ===== Apply RoPE =====
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            
            # ⭐ 关键修复：在 pruning 模式下，需要分割 freqs
            if should_prune and L_denoise is not None:
                # img_query 只有去噪部分，需要对应的 freqs
                # img_freqs 是完整的，需要只取前 L_denoise 部分
                seq_len_query = img_query.shape[1]  # 去噪 tokens 长度
                img_freqs_for_query = img_freqs[:seq_len_query]  # 只取对应部分
                
                # img_key 是完整的（包含去噪+图像），使用完整 freqs
                img_query = apply_rotary_emb_qwen(img_query, img_freqs_for_query, use_real=False)
                img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            else:
                # 正常模式：完整应用
                img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
                img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
        
        # ===== Concatenate for joint attention =====
        # 注意：img_query 可能比 img_key, img_value 短（pruning 时）
        joint_query = torch.cat([txt_query, img_query], dim=1)  # 可能缺少 image query
        joint_key = torch.cat([txt_key, img_key], dim=1)        # 完整的 key
        joint_value = torch.cat([txt_value, img_value], dim=1)  # 完整的 value
        
        # ===== Attention mask 处理 =====
        if encoder_hidden_states_mask is not None:
            # 为 text tokens 创建 mask
            batch_size = joint_query.shape[0]
            num_heads = joint_query.shape[2]
            seq_len_q = joint_query.shape[1]
            seq_len_kv = joint_key.shape[1]
            
            # 创建 attention mask: [B, 1, seq_q, seq_kv]
            attention_mask = torch.zeros(
                (batch_size, 1, seq_len_q, seq_len_kv),
                dtype=joint_query.dtype,
                device=joint_query.device
            )
            
            # Text tokens 的 mask（只 mask text 部分）
            text_mask = encoder_hidden_states_mask.bool()  # [B, L_txt]
            # 扩展到所有 query positions
            text_mask_expanded = text_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_txt]
            text_mask_expanded = text_mask_expanded.expand(batch_size, 1, seq_len_q, seq_txt)
            
            # 应用 mask（将 padding 位置设为 -inf）
            attention_mask[:, :, :, :seq_txt] = attention_mask[:, :, :, :seq_txt].masked_fill(
                ~text_mask_expanded, float('-inf')
            )
        
        # ===== Compute joint attention =====
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        
        # ===== Reshape back =====
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
        
        # ===== Split attention outputs =====
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # 只包含去噪部分（pruning时）
        
        # ===== Output projections =====
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        
        txt_attn_output = attn.to_add_out(txt_attn_output)
        
        # ===== 返回输出 =====
        # 注意：在 pruning 模式下，img_attn_output 只包含去噪部分
        # Block 层面会处理拼接缓存的 image tokens
        return img_attn_output, txt_attn_output


class PrunableQwenImageTransformerBlock(nn.Module):
    """
    支持 Token Pruning 的 Transformer Block
    
    从 QwenImageTransformerBlock 复制并修改
    """
    
    def __init__(self, original_block, layer_idx):
        super().__init__()
        # 复制原始 block 的所有组件
        self.dim = original_block.dim
        self.num_attention_heads = original_block.num_attention_heads
        self.attention_head_dim = original_block.attention_head_dim
        self.layer_idx = layer_idx
        
        # 复制模块引用（共享权重）
        self.img_mod = original_block.img_mod
        self.img_norm1 = original_block.img_norm1
        self.attn = original_block.attn
        self.img_norm2 = original_block.img_norm2
        self.img_mlp = original_block.img_mlp
        
        self.txt_mod = original_block.txt_mod
        self.txt_norm1 = original_block.txt_norm1
        self.txt_norm2 = original_block.txt_norm2
        self.txt_mlp = original_block.txt_mlp
        
        # 替换注意力处理器为支持 pruning 的版本
        self.attn.processor = PrunableQwenDoubleStreamAttnProcessor()
        
        # ⭐ 关键：设置 layer_idx 到 attn，以便 processor 可以访问缓存
        self.attn._layer_idx = layer_idx
    
    def _modulate(self, x, mod_params):
        """Apply modulation to input tensor"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带 Token Pruning 的 forward pass
        """
        # 检查是否需要 pruning
        should_prune = global_pruning_cache.should_prune_current_step()
        L_denoise = global_pruning_cache.denoise_token_length
        
        # ===== 获取缓存（如果需要）=====
        cached_image_k, cached_image_v, cached_image_hidden = None, None, None
        if should_prune and L_denoise is not None:
            cached_image_k, cached_image_v, cached_image_hidden = global_pruning_cache.get_cached_layer_kv(self.layer_idx)
        
        # ===== Modulation parameters =====
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)
        
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        
        # ===== Image stream - norm1 + modulation =====
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
        
        # ===== Text stream - norm1 + modulation =====
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
        
        # ===== Attention computation =====
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        
        img_attn_output, txt_attn_output = attn_output
        
        # ===== 处理 attention 输出（考虑 pruning）=====
        if should_prune and cached_image_hidden is not None and L_denoise is not None:
            # ⚡ Pruning 模式：
            # - img_attn_output 只包含去噪部分
            # - image 部分使用缓存的 hidden states（不更新）
            
            # 去噪部分：应用 attention 更新
            denoise_updated = hidden_states[:, :L_denoise] + img_gate1 * img_attn_output
            
            # 确保缓存在 GPU 上
            if not cached_image_hidden.is_cuda:
                cached_image_hidden = cached_image_hidden.cuda()
            
            # ⚡ 合并：去噪部分（更新） + 图像部分（缓存）
            hidden_states = torch.cat([denoise_updated, cached_image_hidden], dim=1)
        else:
            # 正常模式：完整更新
            hidden_states = hidden_states + img_gate1 * img_attn_output
        
        # 文本流始终正常更新
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        
        # ===== Image stream - norm2 + MLP =====
        if should_prune and cached_image_hidden is not None and L_denoise is not None:
            # ⚡ Pruning 模式：只对去噪 tokens 计算 MLP ⚡
            denoise_hidden = hidden_states[:, :L_denoise]
            denoise_normed2 = self.img_norm2(denoise_hidden)
            denoise_modulated2, denoise_gate2 = self._modulate(denoise_normed2, img_mod2)
            denoise_mlp_output = self.img_mlp(denoise_modulated2)
            denoise_updated = denoise_hidden + denoise_gate2 * denoise_mlp_output
            
            # 确保缓存在 GPU 上
            if not cached_image_hidden.is_cuda:
                cached_image_hidden = cached_image_hidden.cuda()
            
            # ⚡ 合并：去噪部分（更新） + 图像部分（缓存）
            hidden_states = torch.cat([denoise_updated, cached_image_hidden], dim=1)
        else:
            # 正常模式：完整计算
            img_normed2 = self.img_norm2(hidden_states)
            img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
            img_mlp_output = self.img_mlp(img_modulated2)
            hidden_states = hidden_states + img_gate2 * img_mlp_output
        
        # ===== Text stream - norm2 + MLP（始终完整计算）=====
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
        
        # ===== Clip for fp16 =====
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        
        # ⭐ 在 Block 结束时，更新缓存的 hidden states（如果需要）
        # 这样缓存的是经过 Attention + MLP 的最终状态
        if global_pruning_cache.should_cache_current_step() and L_denoise is not None:
            # ⚡ 直接传递 slice，update_layer_hidden 会使用 copy_() 到预分配的 buffer
            image_hidden_final = hidden_states[:, L_denoise:]
            global_pruning_cache.update_layer_hidden(self.layer_idx, image_hidden_final)
        
        return encoder_hidden_states, hidden_states


def apply_token_pruning_to_transformer(transformer):
    """
    将 Transformer 的所有 blocks 替换为支持 pruning 的版本
    
    Args:
        transformer: QwenImageTransformer2DModel 实例
    """
    print(f"\n应用 Token Pruning 到 Transformer ({len(transformer.transformer_blocks)} 层)...")
    
    # 替换所有 transformer blocks
    new_blocks = nn.ModuleList()
    for idx, block in enumerate(transformer.transformer_blocks):
        prunable_block = PrunableQwenImageTransformerBlock(block, layer_idx=idx)
        new_blocks.append(prunable_block)
        if (idx + 1) % 10 == 0:
            print(f"  处理层 {idx + 1}/{len(transformer.transformer_blocks)}")
    
    transformer.transformer_blocks = new_blocks
    print(f"✅ 已替换 {len(new_blocks)} 个 Transformer Blocks")
    
    return transformer

