# -*- coding: utf-8 -*-
import os

# 项目根路径（修改为实际路径）
ROOT_PATH = r"D:\py\works\cat-12"
TEST_IMG_DIR = os.path.join(ROOT_PATH, "cat-12-test")
TEST_LIST_PATH = os.path.join(ROOT_PATH, "test_list")


def generate_test_list():
    """生成test_list（需手动修改标签！）"""
    with open(TEST_LIST_PATH, "w", encoding="utf-8") as f:
        for img_name in os.listdir(TEST_IMG_DIR):
            if img_name.endswith((".jpg", ".png")):
                img_rel_path = f"cat-12-test/{img_name}"
                dummy_label = 0  # 替换为实际标签！！！
                f.write(f"{img_rel_path}\t{dummy_label}\n")

    print(f"✅ 已生成{TEST_LIST_PATH}，请手动修改标签为真实类别（0-11）")


if __name__ == "__main__":
    generate_test_list()