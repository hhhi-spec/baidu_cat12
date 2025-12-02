#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019-10-17
# Created by Author: czliuguoyu@163.com
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from baidu_cat12.CatDataset import Cat
from baidu_cat12.ResNet import ResNet18

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)

epochs = 20
batch_size = 128
learn_rate = 1e-3

data_path = os.path.abspath('data')
file_name = 'train_list.txt'

train_data = Cat(data_path, 224, mode='train', filename=file_name)
val_data = Cat(data_path, 224, mode='val', filename=file_name)
test_data = Cat(data_path, 224, mode='test', filename=file_name)

# 增加num_workers提高数据加载效率
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True)


def evaluate(model, loader):
    '''
    验证模型准确率
    :param model: 模型
    :param loader: 验证数据
    :return acc: 准确率
    '''
    model.eval()
    correct, total = 0, len(loader.dataset)

    for img, label in loader:
        # 将数据移动到GPU
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            logits = model(img)
            predict = logits.argmax(dim=1)

        correct += torch.eq(predict, label).sum().float().item()

    acc = correct / total
    model.train()  # 恢复训练模式
    return acc


def main():
    model = ResNet18(num_class=12)
    # 将模型移动到GPU
    model = model.to(device)

    # 已存在模型参数
    checkpoint_path = os.path.join(os.path.abspath(''), 'cat.cptk')
    if os.path.exists(checkpoint_path):
        # 加载时指定map_location，确保兼容性
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print('Loaded existing model checkpoint.')

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-3)
    # 损失函数
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0

    print(f'Starting training on {device}...')

    for epoch in range(epochs):
        model.train()  # 确保模型在训练模式
        total_loss = 0
        total_batches = 0

        for step, (img, label) in enumerate(train_loader):
            # 将数据移动到GPU
            img, label = img.to(device), label.to(device)

            logits = model(img)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if step % 10 == 0:
                print(f'Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}')

        # 计算平均损失
        avg_loss = total_loss / total_batches if total_batches > 0 else 0

        # 验证
        acc = evaluate(model, val_loader)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'cat.cptk')
            print(f'*** Saving the new model in the local. Best accuracy: {best_acc:.4f} ***')

        print(f'Epoch {epoch}: Accuracy: {acc:.4f}, Best Accuracy: {best_acc:.4f}, Avg Loss: {avg_loss:.4f}')

    print(f'Training completed. Best accuracy: {best_acc:.4f} at epoch {best_epoch}')

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('cat.cptk', map_location=device))
    test_acc = evaluate(model, test_loader)

    print(f'Test accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    main()