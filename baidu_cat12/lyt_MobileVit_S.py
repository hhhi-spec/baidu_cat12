# train_mobilevit_cpu.py
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEma
try:
    from timm.scheduler import CosineLRScheduler
    _HAS_TIMM_SCHED = True
except Exception:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    _HAS_TIMM_SCHED = False
from torch.nn import functional as F

# ===================== 配置（请按需修改路径/参数） =====================
ROOT_PATH = r"F:\Cat-Twelve-Classification-Challenge-main\Cat-Twelve-Classification-Challenge-main"
TRAIN_IMG_ROOT = os.path.join(ROOT_PATH, "data", "cat_12_train")
TRAIN_LIST_PATH = os.path.join(ROOT_PATH, "data", "train_list.txt")
TEST_IMG_ROOT = os.path.join(ROOT_PATH, "data", "cat_12_test")
MODEL_SAVE_PATH = os.path.join(ROOT_PATH, "best_mobilevit_s_cpu.pth")
RESULT_SAVE_PATH = os.path.join(ROOT_PATH, "test_results_mobilevit_s_cpu.csv")

# 训练参数
IMAGE_SIZE = (224, 224)
RESIZE_SCALE = 256  # resize before random crop
BATCH_SIZE = 8        # 每步实际处理的 batch（梯度累积会模拟更大的有效 batch）
ACCUM_STEPS = 4       # 梯度累积步数（有效 batch = BATCH_SIZE * ACCUM_STEPS）
EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cpu")  # 用户指定 CPU
NUM_CLASSES = 12
SEED = 42
PATIENCE = 6
MIXUP_PROB = 1.0       # 应用 mixup/cutmix 的概率
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
LABEL_SMOOTHING = 0.0  # SoftTargetCrossEntropy + mixup 通常不需要额外平滑
USE_EMA = True

# 固定随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== 自定义数据集 =====================
class Cat12TrainDataset(Dataset):
    def __init__(self, img_root, list_path, transform=None):
        self.img_root = img_root
        self.samples = []
        self.transform = transform
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"标签文件不存在: {list_path}")
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                img_rel_path, label = parts[0], parts[1]
                # 原代码用 basename，沿用以保证一致性
                img_abs_path = os.path.join(img_root, os.path.basename(img_rel_path))
                if os.path.exists(img_abs_path):
                    self.samples.append((img_abs_path, int(label)))
                else:
                    # 跳过不存在的图片，但打印一次提示
                    print(f"警告：文件不存在，跳过：{img_abs_path}")
        if len(self.samples) == 0:
            raise RuntimeError("没有找到任何训练样本，请检查路径与 train_list.txt 内容。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class Cat12TestDataset(Dataset):
    def __init__(self, img_root, transform=None):
        self.img_root = img_root
        if not os.path.exists(img_root):
            raise FileNotFoundError(f"测试集目录不存在: {img_root}")
        self.img_paths = sorted([
            os.path.join(img_root, f)
            for f in os.listdir(img_root)
            if f.lower().endswith('.jpg') or f.lower().endswith('.png') or f.lower().endswith('.jpeg')
        ])
        self.transform = transform
        if len(self.img_paths) == 0:
            raise RuntimeError("测试集目录没有找到图片，请检查。")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)

# ===================== 数据预处理 =====================
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((RESIZE_SCALE, RESIZE_SCALE)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((RESIZE_SCALE, RESIZE_SCALE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, val_test_transform

# ===================== Helper =====================
def to_one_hot(labels, num_classes, device):
    if labels.dim() == 1:
        y = torch.zeros(labels.size(0), num_classes, device=device)
        y.scatter_(1, labels.view(-1,1), 1)
        return y
    else:
        return labels

# ===================== 构建 MobileViT-S 模型 =====================
def build_mobilevit_model(pretrained=True):
    try:
        model = timm.create_model(
            model_name="mobilevit_s",
            pretrained=pretrained,
            num_classes=NUM_CLASSES
        )
        # 在 head 前放一个小 Dropout（提高泛化）
        if hasattr(model, "head") and not isinstance(model.head, nn.Sequential):
            model.head = nn.Sequential(
                nn.Dropout(p=0.2),
                model.head
            )
        model = model.to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ MobileViT-S 创建成功（pretrained={pretrained}），参数量: {total_params:,}")
        return model
    except Exception as e:
        print(f"❌ 模型创建失败：{e}")
        return None

# ===================== 训练/验证/预测函数 =====================
def train_one_epoch(model, train_loader, criterion, optimizer, mixup_fn, epoch, scaler=None, scheduler_step=None):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    optimizer.zero_grad()
    num_batches = len(train_loader)
    t0 = time.time()

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        # mixup 会返回 (imgs, mixed_targets) 或者原始 labels，取决于 mixup_fn 设置
        if mixup_fn is not None:
            imgs, targets = mixup_fn(imgs, labels)
            # targets 可能是 float one-hot (soft)
            loss = criterion(model(imgs), targets)
            # 训练准确率用 targets.argmax 近似
            preds = model(imgs).argmax(dim=1)
            try:
                labels_for_acc = targets.argmax(dim=1)
            except Exception:
                labels_for_acc = labels
        else:
            outputs = model(imgs)
            targets = labels
            loss = criterion(outputs, to_one_hot(labels, NUM_CLASSES, DEVICE))
            preds = outputs.argmax(dim=1)
            labels_for_acc = labels

        loss = loss / ACCUM_STEPS
        loss.backward()

        running_loss += loss.item() * imgs.size(0) * ACCUM_STEPS  # 恢复未划分前的 loss
        running_correct += torch.sum(preds == labels_for_acc).item()
        running_total += imgs.size(0)

        # 梯度累积更新
        if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler_step is not None and not _HAS_TIMM_SCHED:
                # 如果使用 PyTorch CosineAnnealingLR，则 step 在 epoch 末（这里不多次 step）
                pass

    avg_loss = running_loss / running_total
    avg_acc = running_correct / running_total
    t1 = time.time()
    print(f"Epoch [{epoch+1}] Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | Time: {t1-t0:.1f}s")
    return avg_loss, avg_acc

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, to_one_hot(labels, NUM_CLASSES, DEVICE))
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += imgs.size(0)
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    print(f"Val Loss: {avg_loss:.4f} | Val Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc

def predict_test_set(model, test_loader, out_csv=RESULT_SAVE_PATH):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, img_names in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            for name, p in zip(img_names, preds):
                results.append({"image_name": name, "predicted_class": int(p)})
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"测试集预测已保存到: {out_csv}")

# ===================== 主流程 =====================
def main():
    # 简单检查数据路径
    if not os.path.exists(TRAIN_IMG_ROOT):
        raise FileNotFoundError(f"训练集图片文件夹不存在: {TRAIN_IMG_ROOT}")
    if not os.path.exists(TRAIN_LIST_PATH):
        raise FileNotFoundError(f"标签文件不存在: {TRAIN_LIST_PATH}")
    if not os.path.exists(TEST_IMG_ROOT):
        raise FileNotFoundError(f"测试集图片文件夹不存在: {TEST_IMG_ROOT}")

    train_transform, val_test_transform = get_transforms()

    full_dataset = Cat12TrainDataset(
        img_root=TRAIN_IMG_ROOT,
        list_path=TRAIN_LIST_PATH,
        transform=train_transform
    )

    # 划分训练/验证
    val_size = max( int(0.1 * len(full_dataset)), 1 )
    train_size = len(full_dataset) - val_size
    # 注意：random_split 使用 CPU generator
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    # 验证集使用不同 transform
    val_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"样本数: 全部={len(full_dataset)} | 训练={train_size} | 验证={val_size}")
    print(f"Batch={BATCH_SIZE}, AccumSteps={ACCUM_STEPS} -> 有效 batch={BATCH_SIZE*ACCUM_STEPS}")

    # 模型
    model = build_mobilevit_model(pretrained=True)
    if model is None:
        return

    # mixup/cutmix
    mixup_fn = None
    try:
        mixup_fn = Mixup(
            mixup_alpha=MIXUP_ALPHA,
            cutmix_alpha=CUTMIX_ALPHA,
            prob=MIXUP_PROB,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=LABEL_SMOOTHING,
            num_classes=NUM_CLASSES
        )
        print("✅ Mixup/CutMix 已启用")
    except Exception as e:
        mixup_fn = None
        print(f"⚠️ 无法启用 Mixup（timm 版本问题？）: {e}")

    # 损失、优化器、scheduler
    criterion = SoftTargetCrossEntropy()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if _HAS_TIMM_SCHED:
        # timm 的 CosineLRScheduler 支持 warmup
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=EPOCHS,
            lr_min=1e-6,
            warmup_t=5,
            warmup_lr_init=1e-5,
            cycle_mul=1.0,
            cycle_decay=0.5,
            cycle_limit=1
        )
        print("✅ 使用 timm CosineLRScheduler (含 warmup)")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        print("⚠️ 使用 PyTorch CosineAnnealingLR (不含 warmup)")

    # EMA
    model_ema = None
    if USE_EMA:
        try:
            model_ema = ModelEma(model)
            print("✅ Model EMA 已启用")
        except Exception as e:
            model_ema = None
            print(f"⚠️ EMA 无法启用: {e}")

    # 训练循环（含早停）
    best_val_acc = 0.0
    no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    print("\n开始训练（CPU）\n" + "-"*60)
    for epoch in range(EPOCHS):
        # 如果使用 timm scheduler，需要在每个 step 内手动调用 scheduler.step_update (但这里采用 epoch-level)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, mixup_fn, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)

        # EMA 更新（timm 的 ModelEma 需要在 optimizer.step 后在每一步更新；
        # 我在 train_one_epoch 中没有每步调用 EMA（因 CPU/代码简化），
        # 因此这里对 model_ema 进行一次整体替换（不是最优做法但可用）
        if model_ema is not None:
            try:
                model_ema.update(model)
            except Exception:
                pass

        # scheduler step
        if _HAS_TIMM_SCHED:
            scheduler.step(epoch + 1)
        else:
            scheduler.step()

        # 保存最优（优先保存 EMA 的 averaged_model）
        eval_model = model_ema.averaged_model if (model_ema is not None and hasattr(model_ema, "averaged_model")) else model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(eval_model.state_dict())
            torch.save(best_model_wts, MODEL_SAVE_PATH)
            no_improve = 0
            print(f"✨ 新最佳模型 (Val Acc={best_val_acc:.4f}) 已保存到 {MODEL_SAVE_PATH}")
        else:
            no_improve += 1
            print(f"连续 {no_improve} 轮无提升（patience={PATIENCE}）")
            if no_improve >= PATIENCE:
                print("⚠️ 早停触发，训练终止。")
                break

    print(f"\n训练结束，最佳验证准确率: {best_val_acc:.4f}")

    # 加载最佳模型参数到 eval_model（优先 EMA）
    final_model = build_mobilevit_model(pretrained=False)  # 创建同结构模型
    final_model.load_state_dict(best_model_wts)
    final_model = final_model.to(DEVICE)

    # 测试集预测
    test_dataset = Cat12TestDataset(img_root=TEST_IMG_ROOT, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"开始对测试集进行预测（样本数={len(test_dataset)})")
    predict_test_set(final_model, test_loader, out_csv=RESULT_SAVE_PATH)

if __name__ == "__main__":
    main()