import os
from PIL import Image
import shutil
from pathlib import Path
import sys

def resize_with_padding(img, target_size, fill_color=(0, 0, 0)):
    """
    核心预处理函数：等比例缩放并填充为正方形。
    """
    width, height = img.size
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    new_img = Image.new('RGB', (target_size, target_size), fill_color)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def preprocess_dataset(source_root, target_root, target_size=512, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    预处理整个数据集。
    
    Args:
        source_root: 原始数据集根目录 (e.g., 'G:/datasetsai')
        target_root: 预处理后数据保存根目录 (e.g., 'G:/datasetsai_processed')
        target_size: 目标图像尺寸
    """
    source_root = Path(source_root)
    target_root = Path(target_root)
    
    # 遍历源目录
    for split_dir in ['train', 'test']:  # 处理训练集和测试集
        split_source = source_root / split_dir
        split_target = target_root / split_dir
        
        if not split_source.exists():
            print(f"警告: 源目录 {split_source} 不存在，跳过。")
            continue
        
        # 对于训练集，遍历类别文件夹；对于测试集，直接处理图片
        if split_dir == 'train':
            # 训练集：预期有 '0_real', '1_fake' 等子文件夹
            class_dirs = [d for d in split_source.iterdir() if d.is_dir()]
            for class_dir in class_dirs:
                class_name = class_dir.name
                process_folder(class_dir, split_target / class_name, target_size, extensions)
        else:
            # 测试集：直接处理图片，不分类
            process_folder(split_source, split_target, target_size, extensions, is_test=True)

def process_folder(source_folder, target_folder, target_size, extensions, is_test=False):
    """处理单个文件夹内的所有图片"""
    target_folder.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    for ext in extensions:
        image_files.extend(source_folder.glob(f'*{ext}'))
        image_files.extend(source_folder.glob(f'*{ext.upper()}'))
    
    print(f"处理文件夹 [{source_folder}]，找到 {len(image_files)} 张图片。")
    
    for i, img_path in enumerate(image_files):
        try:
            # 打开并转换图片
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                processed_img = resize_with_padding(img, target_size)
            
            # 构建目标路径
            if is_test:
                # 测试集：保持原文件名
                target_path = target_folder / img_path.name
            else:
                # 训练集：保持原文件名，但在目标类别文件夹下
                target_path = target_folder / img_path.name
            
            # 保存图片（建议使用高质量JPEG以节省空间）
            processed_img.save(target_path, 'JPEG', quality=95)
            
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(image_files)} 张图片")
                
        except Exception as e:
            print(f"  处理图片 {img_path} 时出错: {e}")
    
    print(f"  完成！图片已保存至 {target_folder}")

if __name__ == '__main__':
    # 配置路径
    SOURCE_DATASET = r'G:/datasetsai'  # 你的原始数据集路径
    TARGET_DATASET = r'G:/datasetsai_processed'  # 预处理后数据保存路径
    TARGET_SIZE = 512  # 统一的目标尺寸
    
    print("开始预处理数据集...")
    print(f"源目录: {SOURCE_DATASET}")
    print(f"目标目录: {TARGET_DATASET}")
    print(f"目标尺寸: {TARGET_SIZE}x{TARGET_SIZE}")
    print("-" * 50)
    
    # 执行预处理
    preprocess_dataset(SOURCE_DATASET, TARGET_DATASET, TARGET_SIZE)
    
    print("\n" + "=" * 50)
    print("数据集预处理全部完成！")
    print(f"预处理后的数据已保存至: {TARGET_DATASET}")
    print("你现在可以使用新的数据路径进行训练了。")