# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from cvnets.models import get_model

# 项目根路径
ROOT_PATH = r"D:\py\works\cat-12"

# 路径配置
BEST_MODEL_PATH = os.path.join(ROOT_PATH, "checkpoints\best_model.pth")
CONFUSION_MATRIX_PATH = os.path.join(ROOT_PATH, "confusion_matrix.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_data():
    """加载模型和测试数据集"""
    # 加载模型
    model_config = {
        "model": "classifier",
        "name": "mobilevit",
        "variant": "s",
        "num_classes": 12,
        "pretrained": False
    }
    model = get_model(model_config).to(DEVICE)

    # 加载最佳模型权重
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    print(f"✅ 加载模型：{BEST_MODEL_PATH}，训练准确率：{checkpoint['acc']:.4f}")

    # 加载测试数据集
    from src.data_preprocess import build_dataset
    _, test_loader, breed_list, _ = build_dataset()
    return model, test_loader, breed_list


def test_model():
    model, test_loader, breed_list = load_model_and_data()
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="测试中")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n总体准确率：{overall_acc:.4f}")
    print("\n分类报告：")
    print(classification_report(
        all_labels, all_preds, target_names=breed_list, digits=4
    ))

    # 混淆矩阵可视化
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=breed_list, yticklabels=breed_list
    )
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.title("十二类猫分类混淆矩阵", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    print(f"\n混淆矩阵已保存至：{CONFUSION_MATRIX_PATH}")
    plt.show()


if __name__ == "__main__":
    test_model()