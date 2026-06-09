"""
模型评估脚本，用于分析模型性能并检查过拟合
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from data_loader1 import create_data_loaders
from train_evaluate import create_model

def evaluate_model(model_path='best_model.pth', 
                   train_root=r"G:\datasetsai_processed\train",
                   test_root=r"G:\datasetsai_processed\test"):
    """
    全面评估模型性能
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return None
    
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 2. 创建数据加载器
    print("\n创建数据加载器...")
    try:
        # 注意：这里使用val_split=0.2来获取验证集
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            train_root=train_root,
            test_root=test_root,
            batch_size=32,
            num_workers=4,
            val_split=0.2,  # 获取验证集
            use_preprocessed=True,
            seed=42  # 确保可重复性
        )
        print(f"类别名称: {class_names}")
        print(f"训练集批次: {len(train_loader)}")
        print(f"验证集批次: {len(val_loader)}")
        print(f"测试集批次: {len(test_loader)}")
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 3. 创建并加载模型
    print("\n初始化模型...")
    try:
        model = create_model(num_classes=len(class_names), device=device, dropout_rate=0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 4. 检查训练历史
    print(f"\n模型检查点信息:")
    for key in checkpoint.keys():
        if key not in ['model_state_dict', 'optimizer_state_dict', 'class_names']:
            print(f"  {key}: {checkpoint[key] if not isinstance(checkpoint[key], torch.Tensor) else checkpoint[key].item()}")
    
    # 5. 在训练集上评估
    print("\n" + "="*50)
    print("在训练集上评估...")
    train_preds, train_labels, train_probs = get_predictions(model, train_loader, device)
    train_accuracy = np.mean(train_preds == train_labels) * 100
    print(f"训练集准确率: {train_accuracy:.2f}% (样本数: {len(train_labels)})")
    
    # 6. 在验证集上评估
    print("\n" + "="*50)
    print("在验证集上评估...")
    val_preds, val_labels, val_probs = get_predictions(model, val_loader, device)
    val_accuracy = np.mean(val_preds == val_labels) * 100
    print(f"验证集准确率: {val_accuracy:.2f}% (样本数: {len(val_labels)})")
    
    # 7. 打印详细分类报告
    print("\n训练集分类报告:")
    print(classification_report(train_labels, train_preds, 
                              target_names=class_names, digits=4))
    
    print("\n验证集分类报告:")
    print(classification_report(val_labels, val_preds, 
                              target_names=class_names, digits=4))
    
    # 8. 生成所有可视化图表
    print("\n" + "="*50)
    print("生成可视化图表...")
    
    # 8.1 绘制混淆矩阵
    try:
        print("生成混淆矩阵...")
        plot_confusion_matrix(val_labels, val_preds, class_names, 
                            title='验证集混淆矩阵')
    except Exception as e:
        print(f"生成混淆矩阵失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 8.2 绘制ROC曲线（仅二分类）
    try:
        print("生成ROC曲线...")
        if len(class_names) == 2:
            plot_roc_curve(val_labels, val_probs[:, 1], class_names)
        else:
            print(f"ROC曲线仅适用于二分类任务，当前为{len(class_names)}分类")
    except Exception as e:
        print(f"生成ROC曲线失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 8.3 绘制精确率-召回率曲线（仅二分类）
    try:
        print("生成精确率-召回率曲线...")
        if len(class_names) == 2:
            plot_precision_recall_curve(val_labels, val_probs[:, 1], class_names)
        else:
            print(f"PR曲线仅适用于二分类任务，当前为{len(class_names)}分类")
    except Exception as e:
        print(f"生成PR曲线失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 8.4 绘制置信度分布图
    try:
        print("生成置信度分布图...")
        plot_confidence_distribution(train_probs, val_probs, class_names)
    except Exception as e:
        print(f"生成置信度分布图失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 9. 过拟合分析
    print("\n" + "="*50)
    print("过拟合分析:")
    print(f"训练准确率: {train_accuracy:.2f}%")
    print(f"验证准确率: {val_accuracy:.2f}%")
    accuracy_gap = abs(train_accuracy - val_accuracy)
    print(f"准确率差距: {accuracy_gap:.2f}%")
    
    if accuracy_gap > 10:
        print("⚠️  警告：可能存在严重过拟合！")
    elif accuracy_gap > 5:
        print("⚠️  警告：可能存在过拟合！")
    else:
        print("✅ 模型泛化能力良好")
    
    # 10. 对测试集进行预测（无标签）
    print("\n" + "="*50)
    print("对测试集进行预测...")
    try:
        # 测试集没有标签，使用专门的函数处理
        test_predictions = predict_test_set_only(model, test_loader, device)
        
        # 保存测试集预测结果
        save_test_predictions_only(test_predictions, class_names)
        
    except Exception as e:
        print(f"测试集预测失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("评估完成!")
    print("生成的文件:")
    for file in ['confusion_matrix.png', 'roc_curve.png', 
                 'precision_recall_curve.png', 'confidence_distribution.png',
                 'test_predictions.csv']:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (未生成)")
    
    return model

def get_predictions(model, data_loader, device):
    """
    获取模型预测结果（适用于有标签的数据）
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 训练集和验证集返回 (images, labels)
            images, labels = batch  # 这里假设batch是(images, labels)
            
            # 确保labels是tensor
            if isinstance(labels, list):
                labels = torch.tensor(labels)
            
            images = images.to(device)
            outputs = model(images)
            
            # 获取概率和预测
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def predict_test_set_only(model, test_loader, device):
    """
    仅对测试集进行预测（没有标签）
    """
    model.eval()
    all_predictions = []
    all_filenames = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 测试集返回 (images, filenames)
            images, filenames = batch
            
            images = images.to(device)
            outputs = model(images)
            
            # 获取概率和预测
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_filenames.extend(filenames)
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return {
        'filenames': all_filenames,
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities)
    }

def save_test_predictions_only(test_predictions, class_names):
    """
    保存测试集预测结果到CSV
    """
    results_df = pd.DataFrame({
        'filename': test_predictions['filenames'],
        'prediction': test_predictions['predictions'],
        'pred_label': [class_names[p] for p in test_predictions['predictions']],
        'max_probability': test_predictions['probabilities'].max(axis=1)
    })
    
    # 添加每个类别的概率
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = test_predictions['probabilities'][:, i]
    
    # 按置信度排序
    results_df = results_df.sort_values('max_probability', ascending=False)
    
    csv_path = 'test_predictions.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"测试集预测结果已保存至: {csv_path}")
    print(f"预测分布:\n{results_df['pred_label'].value_counts()}")
    
    return results_df

def plot_confusion_matrix(y_true, y_pred, class_names, title='混淆矩阵'):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    # 保存图片
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存至: {save_path}")

def plot_roc_curve(y_true, y_scores, class_names):
    """
    绘制ROC曲线（仅二分类）
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = 'roc_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC曲线已保存至: {save_path}, AUC: {roc_auc:.3f}")

def plot_precision_recall_curve(y_true, y_scores, class_names):
    """
    绘制精确率-召回率曲线（仅二分类）
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='PR曲线')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = 'precision_recall_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"精确率-召回率曲线已保存至: {save_path}")

def plot_confidence_distribution(train_probs, val_probs, class_names):
    """
    绘制预测置信度分布
    """
    plt.figure(figsize=(12, 5))
    
    # 训练集置信度分布
    plt.subplot(1, 2, 1)
    train_confidences = train_probs.max(axis=1)
    plt.hist(train_confidences, bins=30, alpha=0.7, color='blue', edgecolor='black', label='训练集')
    plt.xlabel('预测置信度')
    plt.ylabel('样本数量')
    plt.title('训练集预测置信度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.text(0.05, 0.95, f'均值: {train_confidences.mean():.3f}\n标准差: {train_confidences.std():.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 验证集置信度分布
    plt.subplot(1, 2, 2)
    val_confidences = val_probs.max(axis=1)
    plt.hist(val_confidences, bins=30, alpha=0.7, color='orange', edgecolor='black', label='验证集')
    plt.xlabel('预测置信度')
    plt.ylabel('样本数量')
    plt.title('验证集预测置信度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.text(0.05, 0.95, f'均值: {val_confidences.mean():.3f}\n标准差: {val_confidences.std():.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = 'confidence_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"置信度分布图已保存至: {save_path}")

def plot_training_history(checkpoint):
    """
    绘制训练历史（如果可用）
    """
    # 这个函数需要训练历史数据，如果checkpoint中没有则不绘制
    pass

if __name__ == '__main__':
    # 添加命令行参数支持
    import argparse
    
    parser = argparse.ArgumentParser(description='评估AI图片检测模型')
    parser.add_argument('--model', type=str, default='best_model.pth', help='模型文件路径')
    parser.add_argument('--train_root', type=str, default=r"G:\datasetsai_processed\train", help='训练集根目录')
    parser.add_argument('--test_root', type=str, default=r"G:\datasetsai_processed\test", help='测试集根目录')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        train_root=args.train_root,
        test_root=args.test_root
    )