#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019-10-17
# Created by Author: czliuguoyu@163.com
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from baidu_cat12.CatDataset import Cat
from baidu_cat12.ResNet import Flatten
from torchvision.models import resnet18, ResNet18_Weights


# GPU设备配置
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数调整（核心优化）
epochs = 60  # 保持30轮最大训练量
batch_size = 32
learn_rate = 5e-4  # 降低学习率，避免震荡
patience = 12  # 早停耐心值从5→10，允许更多轮无提升

# 数据路径
data_path = os.path.abspath('data')
file_name = 'train_list.txt'

# 构建数据集（后续建议加数据增强，这里先保持不变）
train_data = Cat(data_path, 224, mode='train', filename=file_name)
val_data = Cat(data_path, 224, mode='val', filename=file_name)
test_data = Cat(data_path, 224, mode='test', filename=file_name)

# 数据加载器（num_workers=4报错则改回0）
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, len(loader.dataset)
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            predict = logits.argmax(dim=1)
            correct += torch.eq(predict, label).sum().float().item()
    acc = correct / total
    model.train()
    return acc


def main():
    # 随机初始化ResNet18 + Dropout抑制过拟合
    model = nn.Sequential(
        *list(resnet18(weights=None).children())[:-1],
        Flatten(),
        nn.Dropout(0.6),  # 丢弃60%，防止过拟合
        nn.Linear(512, 12)
    ).to(device)

    model_path = os.path.join(os.path.abspath(''), 'cat_scratch_v2.cptk')
    best_acc, best_epoch = 0.0, 0
    no_improve = 0

    # 优化器 + 学习率衰减（核心优化）
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criteon = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)  # 每8轮学习率减半

    print("Start training from scratch (v2: longer patience + lr decay + dropout)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)

            logits = model(img)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Step [{step}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        # 每轮结束：学习率衰减 + 验证
        scheduler.step()  # 学习率更新
        avg_loss = total_loss / (step + 1)
        val_acc = evaluate(model, val_loader)

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print(f"✅ Updated best model! Epoch: {epoch}, Best Acc: {best_acc:.4f}, Avg Loss: {avg_loss:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            print(f'Epoch [{epoch}/{epochs}], Val Acc: {val_acc:.4f}, Best Acc: {best_acc:.4f}, Avg Loss: {avg_loss:.4f}, No improve: {no_improve}/{patience}')

        # 早停判断（10轮无提升才停）
        if no_improve >= patience:
            print(f"Early stopping! No improvement for {patience} epochs.")
            break

    # 最终测试
    print(f'\n📊 Training finished! Best Val Acc: {best_acc:.4f} at Epoch {best_epoch}')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    main()