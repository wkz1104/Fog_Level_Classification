import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import FogVisibilityModel
from PIL import ImageFile

# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 数据预处理
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=100, device='cuda', patience=10):
    best_val_acc = 0.0
    best_val_loss = np.inf
    best_model_weights = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    counter = 0
    num_classes = len(train_loader.dataset.classes)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # ---------- 训练 ----------
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs, _ = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # ---------- 验证 ----------
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        # 每个类别准确率
        for idx, class_name in enumerate(val_loader.dataset.classes):
            if class_total[idx] > 0:
                acc = 100.0 * class_correct[idx] / class_total[idx]
                print(f'  {class_name} Acc: {acc:.2f}% ({class_correct[idx]}/{class_total[idx]})')
            else:
                print(f'  {class_name} Acc: N/A (0 samples)')

        # ---------- 学习率调度 ----------
        scheduler.step(val_epoch_loss)

        # ---------- 保存最佳模型 ----------
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_val_loss = val_epoch_loss
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, 'best_model.pth')
            print(f'Best model saved with Val Acc: {best_val_acc:.4f}')
            counter = 0
        else:
            if val_epoch_loss > best_val_loss:
                counter += 1
                print(f'EarlyStopping counter: {counter} out of {patience}')
                if counter >= patience:
                    print('Early stopping!')
                    model.load_state_dict(best_model_weights)
                    return model, history
        print()

    model.load_state_dict(best_model_weights)
    return model, history


# 绘制训练曲线
def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.close()


# 主流程
def main():
    data_dir = r'D:\bishe\fog_three\fog_dataset_three'
    batch_size = 16
    num_epochs = 50
    lr = 5e-5
    early_stopping_patience = 6
    light_fog_boost = 3.0  # 轻雾权重加成倍数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据集
    train_dataset = ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=get_transforms(train=True)
    )
    val_dataset = ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=get_transforms(train=False)
    )

    print(f'Class names: {train_dataset.classes}')
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    # 类别权重计算 + 轻雾加权
    num_classes = len(train_dataset.classes)
    class_counts = np.bincount(train_dataset.targets, minlength=num_classes)
    class_counts = np.array([c if c > 0 else 1 for c in class_counts])
    raw_weights = len(train_dataset.targets) / (num_classes * class_counts)
    norm_weights = raw_weights / np.mean(raw_weights)
    norm_weights[1] *= light_fog_boost  # 轻雾加权
    class_weights = torch.FloatTensor(norm_weights).to(device)

    print(f'类别样本数: {class_counts}')
    print(f'类别权重(轻雾加权): {class_weights}')

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型
    model = FogVisibilityModel(num_classes=num_classes).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 训练
    print('Starting training...')
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=num_epochs, device=device, patience=early_stopping_patience
    )

    # 绘图
    plot_training_curves(history)
    print('Training completed!')


if __name__ == '__main__':
    main()
