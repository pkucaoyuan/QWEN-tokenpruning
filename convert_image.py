"""
将 input.jpg 转换为 input.png
"""
from PIL import Image

print("正在转换图像格式...")
print("输入: input.jpg")
print("输出: input.png")

# 读取 JPG
image = Image.open("input.jpg").convert("RGB")

# 保存为 PNG
image.save("input.png")

print(f"✅ 转换完成！")
print(f"   图像尺寸: {image.size}")
print(f"   格式: PNG")

