# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cvnets.models import get_model
from cvnets.utils import load_pretrained_model

# 项目根路径
ROOT_PATH = r"D:\py\works\cat-12"

# 路径配置
PRETRAINED_PATH = os.path.join(ROOT_PATH, "pretrained\mobilevit_s.pt")
CHECKPOINT_DIR = os.path.join(ROOT_PATH, "checkpoints")
LOG_DIR = os.path.join(ROOT_PATH, "logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练超参数
EPOCHS = 300
INIT_LR = 2e-4
MAX_LR = 2e-3
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-2
EMA_DECAY = 0.999


def init_model():
    """初始化MobileViT-S模型"""
    model_config = {
        "model": "classifier",
        "name": "mobilevit",
        "variant": "s",
        "num_classes": 12,
        "pretrained": False
    }
    model = get_model(model_config).to(DEVICE)

    # 加载预训练权重
    if os.path.exists(PRETRAINED_PATH):
        load_pretrained_model(model, PRETRAINED_PATH, ignore_mismatch=True)
        print(f"✅ 加载预训练权重：{PRETRAINED_PATH}")
    else:
        raise FileNotFoundError(f"请下载mobilevit_s.pt到{PRETRAINED_PATH}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=INIT_LR)

    # EMA
    class EMA:
        def __init__(self, model, decay=EMA_DECAY):
            self.model = model
            self.decay = decay
            self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

        def update(self):
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

        def apply(self):
            self.origin = {n: p.data.clone() for n, p in self.model.named_parameters()}
            for n, p in self.model.named_parameters():
                p.data = self.shadow[n]

        def restore(self):
            for n, p in self.model.named_parameters():
                p.data = self.origin[n]

    ema = EMA(model)

    writer = SummaryWriter(LOG_DIR)
    return model, criterion, optimizer, scheduler, ema, writer


def train_epoch(model, criterion, optimizer, scheduler, ema, train_loader, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    # 学习率升温
    if epoch < WARMUP_EPOCHS:
        lr = INIT_LR + (MAX_LR - INIT_LR) * (epoch / WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    else:
        scheduler.step()

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] (Train)")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{avg_acc:.4f}"})

    return avg_loss, avg_acc


def test_epoch(model, criterion, test_loader, ema):
    ema.apply()
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Epoch [Test]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{avg_acc:.4f}"})

    ema.restore()
    return avg_loss, avg_acc


if __name__ == "__main__":
    # 加载数据集
    from src.data_preprocess import build_dataset

    train_loader, test_loader, _, _ = build_dataset()

    # 初始化模型
    model, criterion, optimizer, scheduler, ema, writer = init_model()

    best_acc = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, criterion, optimizer, scheduler, ema, train_loader, epoch)
        test_loss, test_acc = test_epoch(model, criterion, test_loader, ema)

        # 记录日志
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/Acc", test_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "acc": test_acc
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"✅ 保存最佳模型，Acc: {best_acc:.4f}")

    writer.close()
    print(f"训练完成，最佳测试准确率：{best_acc:.4f}")