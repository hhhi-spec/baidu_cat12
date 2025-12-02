import os
import torch
import timm
from timm.models import create_model

# 项目根路径
ROOT_PATH = r"D:\py\works\cat-12"
PRETRAINED_PATH = os.path.join(ROOT_PATH, "pretrained", "mobilevit_s.pt")

def download_mobilevit_s():
    os.makedirs(os.path.dirname(PRETRAINED_PATH), exist_ok=True)
    # 增加超时时间（单位：秒）
    model = create_model('mobilevit_s.cvnets_in1k', pretrained=True, timeout=60)
    torch.save(model.state_dict(), PRETRAINED_PATH)
    print(f"✅ 预训练权重已保存至：{PRETRAINED_PATH}")

if __name__ == "__main__":
    download_mobilevit_s()