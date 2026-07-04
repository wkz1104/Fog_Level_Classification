import os
import random
from PIL import Image
import numpy as np

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)

def load_image(image_path):
    """加载图片，处理可能的异常"""
    try:
        img = Image.open(image_path).convert('RGB')  # 统一转为RGB格式，避免灰度图问题
        return img
    except Exception as e:
        print(f"加载图片 {image_path} 失败: {e}")
        return None

def random_horizontal_flip(img, p=0.5):
    """随机水平翻转"""
    if random.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def random_rotation(img, degrees=10):
    """随机旋转"""
    angle = random.uniform(-degrees, degrees)
    return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))  # 背景填充白色

def random_resized_crop(img, size, scale=(0.8, 1.0)):
    """随机裁剪后resize"""
    width, height = img.size
    # 计算裁剪区域的大小
    area = width * height
    target_area = random.uniform(scale[0], scale[1]) * area
    # 计算裁剪的宽高比（接近原图比例）
    aspect_ratio = random.uniform(0.8, 1.2)  # 宽高比范围
    
    w = int(round(np.sqrt(target_area * aspect_ratio)))
    h = int(round(np.sqrt(target_area / aspect_ratio)))
    
    # 确保裁剪区域不超出原图
    w = min(w, width)
    h = min(h, height)
    
    # 随机选择裁剪位置
    x1 = random.randint(0, width - w)
    y1 = random.randint(0, height - h)
    
    # 裁剪并resize
    img_crop = img.crop((x1, y1, x1 + w, y1 + h))
    return img_crop.resize((size, size), Image.Resampling.BILINEAR)

def color_jitter(img, brightness=0.3, contrast=0.3):
    """随机调整亮度和对比度"""
    # 调整亮度
    if brightness > 0:
        brightness_factor = random.uniform(1 - brightness, 1 + brightness)
        img = img.point(lambda p: p * brightness_factor)
    
    # 调整对比度
    if contrast > 0:
        contrast_factor = random.uniform(1 - contrast, 1 + contrast)
        img = img.point(lambda p: p * contrast_factor)
    
    # 确保像素值在0-255范围内
    img = np.clip(np.array(img), 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def random_grayscale(img, p=0.1):
    """随机转为灰度图"""
    if random.random() < p:
        img_gray = img.convert('L')
        # 转回RGB格式（3通道）
        return Image.merge('RGB', (img_gray, img_gray, img_gray))
    return img

def normalize(img):
    """标准化（模拟PyTorch的Normalize，最后转回0-255）"""
    # ImageNet均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # 转为numpy数组并归一化到[0,1]
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # 标准化
    for i in range(3):
        img_np[:, :, i] = (img_np[:, :, i] - mean[i]) / std[i]
    
    # 反标准化转回0-255（方便保存）
    img_np = (img_np * std + mean) * 255
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_np)

def augment_image(img, size=384):
    """执行完整的增强流程"""
    if img is None:
        return None
    
    # 1. Resize
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    
    # 2. 随机水平翻转
    img = random_horizontal_flip(img)
    
    # 3. 随机旋转
    img = random_rotation(img)
    
    # 4. 随机裁剪并resize
    img = random_resized_crop(img, size)
    
    # 5. 颜色抖动
    img = color_jitter(img)
    
    # 6. 随机灰度化
    img = random_grayscale(img)
    
    # 7. 标准化（最后转回可视范围）
    img = normalize(img)
    
    return img

def process_folder(input_dir, output_dir, augment_num=1):
    """
    处理文件夹中的所有图片
    :param input_dir: 输入图片文件夹路径
    :param output_dir: 输出增强图片文件夹路径
    :param augment_num: 每张图片生成的增强版本数量
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # 遍历输入文件夹
    for filename in os.listdir(input_dir):
        # 过滤非图片文件
        if not filename.lower().endswith(supported_formats):
            continue
        
        # 拼接完整路径
        img_path = os.path.join(input_dir, filename)
        # 加载图片
        img = load_image(img_path)
        if img is None:
            continue
        
        # 生成多张增强图片
        for i in range(augment_num):
            # 执行增强
            aug_img = augment_image(img)
            if aug_img is None:
                continue
            
            # 生成输出文件名
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_aug_{i+1}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存图片
            try:
                aug_img.save(output_path)
                print(f"已保存增强图片: {output_path}")
            except Exception as e:
                print(f"保存图片 {output_path} 失败: {e}")

if __name__ == "__main__":
    # 配置参数
    INPUT_FOLDER = r"D:\bishe\test"   # 输入文件夹（存放原始图片）
    OUTPUT_FOLDER = "./augmented_images"  # 输出文件夹（存放增强后图片）
    AUGMENT_NUM = 1
    
    # 执行处理
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, AUGMENT_NUM)