# ✍️ SVM 手写数字识别

基于 **SVM（支持向量机）** 的 MNIST 手写数字识别系统，带有美观的 PyQt5 图形界面，支持鼠标手写输入并实时预测。

## 🎯 功能

- 🖌️ **手写输入** — 280×280 绘制画布，鼠标手写数字
- ⚡ **实时预测** — 即时显示识别结果和置信度
- 🎨 **深色科技风 UI** — 现代化的暗色主题界面
- 🔢 **Top-K 预测** — 支持显示前 3 个最可能的数字及概率
- 📊 **完整实验报告** — 训练过程自动生成混淆矩阵、支持向量分析等可视化图表

## 📸 界面截图

| 绘制区域 | 预测结果 |
|----------|----------|
| 深色画布手写输入 | 大字数字 + 置信度进度条 |

## 🚀 快速开始

### 环境要求

- Python ≥ 3.10
- pip

### 安装

```bash
# 克隆仓库
git clone https://github.com/wisemen-dev/DigitRecognition_SVM.git
cd DigitRecognition_SVM

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py
```

训练完成后会在 `model/` 目录生成：
- `svm_mnist.pkl` — 训练好的 SVM 模型
- `scaler.pkl` — 数据标准化器

同时在 `figures/` 目录生成实验可视化图片。

### 启动 GUI

```bash
python main.py
```

在画布上使用鼠标绘制数字（0-9），点击 **识别** 按钮即可查看预测结果。

## 🧠 模型说明

| 项目 | 说明 |
|------|------|
| 数据集 | MNIST（70,000 张 28×28 手写数字图片） |
| 特征提取 | 像素值归一化 + StandardScaler 标准化 |
| 分类器 | SVM（RBF 核） + Platt 概率校准 |
| 输入尺寸 | 28×28 灰度图 → 784 维特征向量 |
| 预处理 | 灰度转换 → 缩放 → 归一化 → 反色 → 标准化 |

## 📁 项目结构

```
DigitRecognition_SVM/
├── main.py              # 程序入口
├── train.py             # 模型训练脚本
├── model.py             # 模型封装（加载、预处理、预测）
├── gui.py               # PyQt5 图形界面
├── requirements.txt     # 依赖列表
├── model/               # 训练好的模型文件
│   ├── svm_mnist.pkl
│   └── scaler.pkl
├── figures/             # 可视化结果图
│   ├── figure_1_mnist_samples.png
│   ├── figure_2_confusion_matrix.png
│   ├── figure_3_support_vectors.png
│   ├── figure_4_parameter_experiments.png
│   └── figure_5_error_samples.png
└── docs/                # 设计文档
```

## 🔧 依赖

| 库 | 用途 |
|----|------|
| `numpy` | 数值计算 |
| `scikit-learn` | SVM 模型、预处理、评估 |
| `scipy` | 科学计算 |
| `PyQt5` | GUI 图形界面 |
| `joblib` | 模型序列化 |
| `Pillow` | 图像处理 |
| `matplotlib` | 可视化图表生成 |

## 📄 License

MIT
