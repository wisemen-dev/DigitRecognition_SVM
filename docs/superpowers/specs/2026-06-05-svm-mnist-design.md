# SVM 手写数字识别系统 — 设计文档

**日期:** 2026-06-05
**状态:** 已批准

---

## 概述

基于 SVM（支持向量机）的手写数字识别系统。使用 MNIST 数据集训练 SVC 模型，并通过 PyQt5 构建美观的独立手写窗口供用户绘制数字并实时识别。

## 架构

```
DigitRecognition_SVM/
├── train.py              # 训练脚本
├── model.py              # 模型封装
├── gui.py                # PyQt5 GUI
├── main.py               # 入口
├── model/
│   └── svm_mnist.pkl     # 持久化模型
└── requirements.txt
```

## 模块设计

### train.py
- 使用 `sklearn.datasets.fetch_openml('mnist_784')` 加载 MNIST
- `StandardScaler` 标准化 + `SVC(kernel='rbf')` 训练
- 评估准确率，保存模型到 `model/svm_mnist.pkl`

### model.py
- `DigitRecognizer` 类：加载模型，提供 `predict(image_28x28)` 方法
- 图像预处理：缩放至 28x28 + 归一化

### gui.py
- `DrawCanvas`: 280x280 黑色画布，鼠标拖动绘制
- `RecognitionWindow`: 主窗口，深色主题，包含画布、预测结果显示、操作按钮
- QSS 样式：深蓝灰背景 + 翠绿/珊瑚色圆角按钮

### main.py
- 检查模型文件是否存在，不存在则提示先运行 train.py
- 加载模型并启动 GUI

## 视觉风格

- 背景: #1a1a2e, 画布边框: #00ff88, 按钮渐变
- 预测结果大字显示 + 置信度进度条
