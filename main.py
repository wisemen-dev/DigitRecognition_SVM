"""
SVM 手写数字识别系统 — 程序入口
================================
检查模型是否存在 → 不存在则提示训练 → 存在则加载并启动 GUI。
"""

import os
import sys


def main():
    model_path = "model/svm_mnist.pkl"
    scaler_path = "model/scaler.pkl"

    # 检查模型和标准化器是否已训练
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("=" * 60)
        print("  ⚠️  未找到训练好的模型！")
        print("=" * 60)
        print(f"\n  需要先运行 train.py 来训练模型。\n")
        print(f"  请执行:  python train.py\n")
        print("=" * 60)
        sys.exit(1)

    # 导入模型和 GUI（放在这里避免启动时不必要的导入延迟）
    from model import DigitRecognizer
    from gui import launch_gui

    print("加载模型中 ...")
    recognizer = DigitRecognizer(model_path, scaler_path)
    print("模型加载完成，启动 GUI ...\n")

    launch_gui(recognizer)


if __name__ == "__main__":
    main()
