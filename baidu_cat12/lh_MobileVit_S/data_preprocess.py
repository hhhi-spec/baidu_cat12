# -*- coding: utf-8 -*-
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# 项目根路径
ROOT_PATH = r"/"

# 数据路径配置
TRAIN_LIST = os.path.join(ROOT_PATH, "train_list.txt")
TEST_LIST = os.path.join(ROOT_PATH, "test_list.txt")
TRAIN_IMG_DIR = os.path.join(ROOT_PATH, "cat-12-train")
TEST_IMG_DIR = os.path.join(ROOT_PATH, "cat-12-test")
INPUT_SIZE = 256  # MobileViT输入尺寸
BATCH_SIZE = 16  # 按GPU显存调整（8GB→8，16GB→16）


def parse_label_file(txt_path, img_root):
    """解析train/test_list.txt，返回图片路径和标签"""
    img_paths, labels = [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_rel, label = line.split("\t")
            img_abs = os.path.join(img_root, os.path.basename(img_rel))
            if os.path.exists(img_abs):
                img_paths.append(img_abs)
                labels.append(int(label))
    return img_paths, labels


class CatDataset(Dataset):
    def __init__(self, img_paths, labels, is_train=True):
        self.img_paths = img_paths
        self.labels = labels
        self.is_train = is_train

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.is_train:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        return img, self.labels[idx]


def build_dataset():
    """构建训练/测试数据集和DataLoader"""
    # 解析训练集
    train_paths, train_labels = parse_label_file(TRAIN_LIST, TRAIN_IMG_DIR)
    # 解析测试集
    test_paths, test_labels = parse_label_file(TEST_LIST, TEST_IMG_DIR)

    # 标签映射（假设0-11对应12类猫）
    breed_list = [f"Cat_{i}" for i in range(12)]  # 可替换为真实猫种名
    label_map = {breed: idx for idx, breed in enumerate(breed_list)}

    # 创建Dataset
    train_dataset = CatDataset(train_paths, train_labels, is_train=True)
    test_dataset = CatDataset(test_paths, test_labels, is_train=False)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"训练集：{len(train_paths)}张，测试集：{len(test_paths)}张")
    return train_loader, test_loader, breed_list, label_map


if __name__ == "__main__":
    train_loader, test_loader, breed_list, label_map = build_dataset()