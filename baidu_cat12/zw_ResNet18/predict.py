#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image
from torchvision import transforms
from baidu_cat12.ResNet import ResNet18  # 导入你的ResNet18模型

# ===================== 1. 基础配置 =====================
# 设备（和训练时保持一致）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 类别映射（需和训练时的标签对应！比如0=品种A，1=品种B...，你需补充实际类别名）
# 示例：cat_classes = ['品种0', '品种1', '品种2', ..., '品种11']
cat_classes = [f'猫咪品种{i}' for i in range(12)]
# 模型路径（训练好的cat.cptk）
model_path = 'cat.cptk'
# 待预测的图片文件夹
test_img_dir = os.path.join('data', 'cat_12_test')
# 图像预处理（必须和训练时CatDataset的预处理一致！）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放到224×224
    transforms.ToTensor(),  # 转为张量（0-1）
    # 以下归一化参数需和训练时一致（通常用ImageNet均值/std）
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ===================== 2. 加载模型 =====================
def load_model():
    # 初始化模型（和训练时一致：12分类）
    model = ResNet18(num_class=12)
    model = model.to(device)
    # 加载训练好的参数（添加weights_only=True消除警告）
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 切换到预测模式（关闭Dropout/BatchNorm）
    return model


# ===================== 3. 单张图片预测 =====================
def predict_single_img(model, img_path):
    # 读取图片（RGB格式）
    img = Image.open(img_path).convert('RGB')
    # 预处理
    img_tensor = transform(img).unsqueeze(0)  # 增加batch维度（模型要求输入是[batch, c, h, w]）
    img_tensor = img_tensor.to(device)

    # 预测
    with torch.no_grad():
        logits = model(img_tensor)
        pred_idx = logits.argmax(dim=1).item()  # 取得分最高的类别索引
        pred_class = cat_classes[pred_idx]  # 转为类别名
        pred_prob = torch.softmax(logits, dim=1)[0][pred_idx].item()  # 预测概率

    return pred_class, pred_prob


# ===================== 4. 批量预测文件夹下所有图片 =====================
def batch_predict():
    model = load_model()
    # 遍历文件夹下所有jpg文件
    for img_name in os.listdir(test_img_dir):
        if not img_name.endswith('.jpg'):
            continue
        img_path = os.path.join(test_img_dir, img_name)
        # 预测
        pred_class, pred_prob = predict_single_img(model, img_path)
        # 打印结果
        print(f'图片：{img_name} → 预测类别：{pred_class} → 置信度：{pred_prob:.4f}')


if __name__ == '__main__':
    batch_predict()