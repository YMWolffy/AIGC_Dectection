import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='YouYuan')
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
# 导入我们自定义的数据加载器
from data_loader1 import create_data_loaders

def create_model(num_classes=2, device='cuda', dropout_rate=0.5):
    """
    创建并初始化模型，添加Dropout层防止过拟合。
    使用在ImageNet上预训练的ResNet-50，并替换最后的全连接层以适应二分类任务。
    """
    # 加载预训练的ResNet-50模型
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # 冻结部分层（只训练最后几层）
    # 冻结前几个卷积块，只微调后面几层
    for name, param in model.named_parameters():
        if 'layer1' in name or 'layer2' in name or 'conv1' in name or 'bn1' in name:
            param.requires_grad = False
    
    # 替换最后的全连接层，添加Dropout
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate/2),
        nn.Linear(256, num_classes)
    )
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """
    训练一个epoch。
    集成了性能分析代码，用于区分数据加载时间和GPU计算时间。
    可选：支持混合精度训练（通过scaler参数）。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # --- 性能分析变量 ---
    batch_load_time = 0.0    # 批次数据加载总时间
    gpu_comp_time = 0.0      # GPU纯计算时间
    analyzed_batches = 0
    
    epoch_start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start_time = time.time()
        
        # 将数据移动到设备
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        
        # 计算数据加载与传输耗时
        data_transfer_end = time.time()
        batch_load_time += (data_transfer_end - batch_start_time)
        
        optimizer.zero_grad()
        
        # --- GPU计算开始 ---
        gpu_start_time = time.time()
        
        if scaler is not None:
            # 使用混合精度训练
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            # 梯度裁剪防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 使用标准精度训练
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        torch.cuda.synchronize()  # 确保GPU计时准确
        gpu_end_time = time.time()
        # --- GPU计算结束 ---
        
        gpu_comp_time += (gpu_end_time - gpu_start_time)
        analyzed_batches += 1
        
        # 计算指标
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 每处理一定批次后打印性能分析
        if (batch_idx + 1) % 50 == 0:
            avg_batch_load = batch_load_time / analyzed_batches
            avg_gpu_comp = gpu_comp_time / analyzed_batches
            gpu_utilization = (avg_gpu_comp / (avg_batch_load + avg_gpu_comp)) * 100 if (avg_batch_load + avg_gpu_comp) > 0 else 0
            
            print(f'[性能分析] Epoch {epoch:03d} | Batch {batch_idx+1:4d} | '
                  f'Loss: {loss.item():.4f} | '
                  f'GPU利用率: {gpu_utilization:.1f}%')
    
    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    print(f'[Epoch {epoch:03d} 总结] 总耗时: {epoch_time:.2f}s | '
          f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    在验证集上进行验证。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def predict_test_set(model, test_loader, device, class_names):
    """
    对无标签的测试集进行预测，并保存结果到CSV文件。
    """
    model.eval()
    all_filenames = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # 获取预测类别和概率
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_filenames.extend(filenames)
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'prediction': all_predictions,
        'pred_label': [class_names[p] for p in all_predictions],
        'prob_fake': [prob[0] for prob in all_probabilities],  # 假设类别0是'fake'
        'prob_real': [prob[1] for prob in all_probabilities]   # 假设类别1是'real'
    })
    
    return results_df

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # ==================== 参数配置 ====================
    TRAIN_ROOT = r"G:\datasetsai_processed\train"
    TEST_ROOT = r"G:\datasetsai_processed\test"
    
    TARGET_SIZE = 512
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_EPOCHS = 10  # 增加epoch，配合早停
    LEARNING_RATE = 1e-4
    
    # 正则化参数
    DROPOUT_RATE = 0.5
    VAL_SPLIT_RATIO = 0.2  # 20%的训练数据作为验证集
    PATIENCE = 5  # 早停耐心值
    USE_AMP = True
    # ================================================
    
    print(f"训练集路径: {TRAIN_ROOT}")
    print(f"测试集路径: {TEST_ROOT}")
    print(f"批次大小: {BATCH_SIZE}, 数据加载进程: {NUM_WORKERS}")
    print(f"验证集比例: {VAL_SPLIT_RATIO}")
    print(f"启用混合精度训练: {USE_AMP}")
    
    # 1. 创建数据加载器（包含验证集）
    print("\n" + "="*50)
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
    train_root=TRAIN_ROOT,
    test_root=TEST_ROOT,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    val_split=VAL_SPLIT_RATIO,  # 确保 data_loader.py 中有这个参数
    use_preprocessed=True
)
    
    # 2. 初始化模型、损失函数、优化器
    print("初始化模型...")
    model = create_model(num_classes=2, device=device, dropout_rate=DROPOUT_RATE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 学习率调度器（使用余弦退火 + 热重启）
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # 混合精度训练的梯度缩放器
    scaler = GradScaler() if (USE_AMP and str(device) == 'cuda') else None
    
    # 3. 训练循环
    print("\n" + "="*50)
    print("开始训练...")
    
    best_val_acc = 0.0
    patience_counter = 0
    train_history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录历史
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        print(f'[Epoch {epoch:03d} 结果] 训练Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | '
              f'验证Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 早停和模型保存逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_names': class_names
            }, 'best_model.pth')
            print(f'==> 已保存最佳模型至 best_model.pth (验证准确率: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            print(f'验证准确率未提升，耐心计数: {patience_counter}/{PATIENCE}')
            
            if patience_counter >= PATIENCE:
                print(f"\n早停触发！在 epoch {epoch} 停止训练")
                break
    
    print("\n" + "="*50)
    print("训练完成！")
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    
    # 4. 加载最佳模型并对测试集进行预测
    print("\n" + "="*50)
    print("使用最佳模型对测试集进行预测...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    results_df = predict_test_set(model, test_loader, device, class_names)
    
    # 保存预测结果
    results_csv_path = 'test_predictions.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"预测结果已保存至: {results_csv_path}")
    
    # 打印预测结果统计
    print(f"\n预测结果分布:")
    print(results_df['pred_label'].value_counts())
    
    # 5. 绘制训练损失和准确率曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='训练损失')
    plt.plot(train_history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_history['acc'], label='训练准确率')
    plt.plot(train_history['val_acc'], label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("训练历史曲线已保存至: training_history.png")

if __name__ == '__main__':
    main()