import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np


class PaddingResize:
    """
    可序列化的自定义Transform类，用于等比例缩放并填充图像。
    """
    def __init__(self, target_size=512, fill_color=(0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color
    
    def __call__(self, img):
        """
        将图像等比例缩放，并用fill_color填充为正方形。
        """
        width, height = img.size
        scale = self.target_size / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        new_img = Image.new('RGB', (self.target_size, self.target_size), self.fill_color)
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img

class UnlabeledTestDataset(Dataset):
    """
    用于加载无标签测试集的自定义Dataset类。
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        
        # 遍历文件夹，收集所有支持的图片文件
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        for ext in valid_extensions:
            self.image_paths.extend(list(self.root_dir.glob(f'*{ext}')))
        
        # 按文件名排序以确保可重复性
        self.image_paths = sorted(self.image_paths)
        print(f"在测试集中找到 {len(self.image_paths)} 张图片。")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 打开并转换图片为RGB
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图片 {img_path} 时出错: {e}")
            # 返回一个占位符图像
            image = Image.new('RGB', (512, 512), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, os.path.basename(img_path)

def get_transform_for_preprocessed(is_train=True, target_size=512):
    """
    为预处理后的图像定义数据转换流程。
    增加更多数据增强以防止过拟合。
    """
    if is_train:
        # 训练集：更强的数据增强
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),  # 新增：随机垂直翻转
            transforms.RandomRotation(degrees=15),  # 新增：随机旋转
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 新增：随机平移
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 增强颜色抖动
            transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),  # 新增：随机裁剪和缩放
            transforms.RandomGrayscale(p=0.1),  # 新增：随机灰度化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # 新增：随机擦除
        ])
    else:
        # 验证集/测试集：仅标准化和中心裁剪
        transform = transforms.Compose([
            transforms.Resize(target_size + 32),  # 稍微放大
            transforms.CenterCrop(target_size),   # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform

def create_data_loaders(train_root, test_root, batch_size=32, num_workers=4, 
                        val_split=0.2, use_preprocessed=True, seed=42):
    """
    Create training, validation and test data loaders.
    
    参数:
        train_root: 训练集根目录
        test_root: 测试集根目录
        batch_size: 每个批次的图片数量
        num_workers: 用于数据加载的子进程数量
        val_split: 验证集比例
        use_preprocessed: 是否使用预处理后的图像
        seed: 随机种子
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 根据是否使用预处理数据选择合适的转换流程
    if use_preprocessed:
        train_transform = get_transform_for_preprocessed(is_train=True, target_size=512)
        val_transform = get_transform_for_preprocessed(is_train=False, target_size=512)
        test_transform = get_transform_for_preprocessed(is_train=False, target_size=512)
    else:
        train_transform = transforms.Compose([
            PaddingResize(target_size=512),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            PaddingResize(target_size=512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = val_transform
    
    # 加载训练集
    try:
        full_train_dataset = datasets.ImageFolder(root=train_root, transform=train_transform)
        print(f"训练集类别: {full_train_dataset.classes}")
        print(f"总训练样本数: {len(full_train_dataset)}")
        
        # 按类别统计样本数量
        class_counts = {}
        for class_idx in range(len(full_train_dataset.classes)):
            class_counts[full_train_dataset.classes[class_idx]] = sum(
                [1 for _, label in full_train_dataset.samples if label == class_idx]
            )
        print(f"类别分布: {class_counts}")
        
        # 划分训练集和验证集
        val_size = int(val_split * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
        
    except Exception as e:
        print(f"加载训练集失败: {e}")
        print(f"请检查路径是否正确: {train_root}")
        raise
    
    # 测试集：使用自定义的无标签Dataset
    test_dataset = UnlabeledTestDataset(root_dir=test_root, transform=test_transform)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # 丢弃最后一个不完整的批次，保持批次大小一致
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    
    print(f"测试集样本数: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, full_train_dataset.classes

# 简易测试代码
if __name__ == '__main__':
    TRAIN_PATH = r"G:/datasetsai_processed/train"
    TEST_PATH = r"G:/datasetsai_processed/test"
    
    print("测试数据加载器...")
    try:
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            train_root=TRAIN_PATH,
            test_root=TEST_PATH,
            batch_size=16,
            num_workers=2,
            val_split=0.2,
            use_preprocessed=True
        )
        
        print("\n测试训练集加载...")
        for images, labels in train_loader:
            print(f"训练集批次 - 图像形状: {images.shape}, 标签形状: {labels.shape}")
            print(f"标签分布: {torch.bincount(labels)}")
            break
            
        print("\n测试验证集加载...")
        for images, labels in val_loader:
            print(f"验证集批次 - 图像形状: {images.shape}, 标签形状: {labels.shape}")
            break
            
        print("\n测试测试集加载...")
        for images, filenames in test_loader:
            print(f"测试集批次 - 图像形状: {images.shape}, 文件名示例: {filenames[:3]}")
            break
            
        print(f"\n类别名称: {class_names}")
        print("✅ 数据加载器创建成功!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()