# 快速修复：transformers 兼容性错误

## ❌ 常见错误

### 错误1：transformers 兼容性

```
AttributeError: 'dict' object has no attribute 'to_dict'
```

发生在加载 Qwen2.5-VL text_encoder 时。

### 错误2：PEFT backend 缺失

```
ValueError: PEFT backend is required for this method.
```

发生在加载 LoRA 权重时。

---

## ✅ 解决方案

### **方法1：自动修复（推荐）⭐**

运行环境检查和修复脚本：

```bash
python 0_环境检查和修复.py
```

此脚本会：
- ✅ 自动检查所有依赖
- ✅ 自动升级 transformers 到 4.48.0+
- ✅ **自动安装 peft 库**
- ✅ 自动修复环境问题
- ✅ 自动转换图片格式

---

### **方法2：手动修复（快速）**

```bash
# 安装缺失的依赖
pip install --upgrade transformers>=4.48.0
pip install peft>=0.13.0

# 重新运行
python 2_load_and_inference_lightning.py
```

---

### **方法3：使用修复脚本**

```bash
python fix_transformers_error.py
```

---

## 🔧 完整修复流程

```bash
# 1. 升级 transformers
pip install --upgrade transformers>=4.48.0

# 2. 验证版本
python -c "import transformers; print(transformers.__version__)"
# 应该显示 4.48.0 或更高

# 3. 重新运行推理
python 2_load_and_inference_lightning.py
```

---

## 📋 推荐的依赖版本

```
torch>=2.0.0
transformers>=4.48.0  ⭐ 关键
diffusers>=0.36.0
accelerate>=0.20.0
pillow>=9.0.0
safetensors>=0.3.0
huggingface_hub>=0.20.0
peft>=0.13.0  ⭐ LoRA 加载必需
```

---

## 🚀 从头开始（完整流程）

```bash
# 1. 克隆仓库
git clone https://github.com/pkucaoyuan/QWEN-tokenpruning.git
cd QWEN-tokenpruning

# 2. 运行环境检查（自动修复）⭐ 推荐
python 0_环境检查和修复.py

# 3. 运行推理
python 2_load_and_inference_lightning.py
```

---

## 🐛 如果仍然出错

### 错误1：transformers 版本仍然过低

```bash
# 强制重装最新版
pip uninstall transformers -y
pip install transformers>=4.48.0
```

### 错误2：CUDA 不可用

```bash
# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果为 False，检查 PyTorch 安装
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 错误3：显存不足

在脚本中添加：
```python
# 在 pipe.to("cuda") 之后添加
pipe.enable_model_cpu_offload()  # CPU offload
pipe.vae.enable_tiling()         # VAE tiling
```

---

## ✅ 验证修复成功

```bash
# 检查 transformers 版本
python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# 应该显示: transformers: 4.48.0 或更高
```

---

## 📞 如果问题持续

1. 检查 Python 版本：`python --version`（需要 3.8+）
2. 检查虚拟环境：确保在正确的环境中
3. 清理缓存：`pip cache purge`
4. 重新安装依赖：`pip install -r requirements.txt --upgrade`

