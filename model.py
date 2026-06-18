"""
SVM 手写数字识别 — 模型封装
============================
DigitRecognizer 类负责：
- 加载训练好的 SVM 模型和 StandardScaler
- 对输入图像进行预处理（缩放、居中、归一化）
- 返回预测结果和置信度概率
"""

import numpy as np
from PIL import Image
import joblib


class DigitRecognizer:
    """手写数字识别器，封装 SVM 模型加载、预处理和预测。"""

    def __init__(self, model_path: str = "model/svm_mnist.pkl",
                 scaler_path: str = "model/scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self._classes = self.model.classes_

    def preprocess(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """
        将输入图像预处理为模型所需的 784 维特征向量。

        流程：
        1. 转为灰度图
        2. 缩放到 28×28
        3. 转为 numpy 数组并归一化到 [0, 1]
        4. 反色（MNIST 是黑底白字）
        5. 展平为 784 维并标准化
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # 转为灰度
        image = image.convert("L")

        # 缩放到 28×28（使用 LANCZOS 高质量重采样）
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # 转为 numpy 数组 [0, 255] → [0, 1]
        arr = np.array(image, dtype=np.float32) / 255.0

        # 展平为 784 维向量
        arr = arr.reshape(1, -1)

        # StandardScaler 标准化
        arr = self.scaler.transform(arr)

        return arr

    def predict(self, image: Image.Image | np.ndarray) -> tuple[int, float]:
        """
        对输入图像进行预测。

        返回:
            (predicted_digit, confidence): 预测数字和置信度 (0~1)
        """
        features = self.preprocess(image)
        digit = int(self.model.predict(features)[0])

        # 获取概率（需要训练时设置 probability=True）
        proba = self.model.predict_proba(features)[0]
        confidence = float(np.max(proba))

        return digit, confidence

    def predict_top_k(self, image: Image.Image | np.ndarray, k: int = 3
                      ) -> list[tuple[int, float]]:
        """
        返回 Top-K 预测结果。

        返回:
            [(digit, probability), ...] 按概率降序排列
        """
        features = self.preprocess(image)
        proba = self.model.predict_proba(features)[0]

        # 按概率降序排列
        top_indices = np.argsort(proba)[::-1][:k]
        results = [(int(self._classes[i]), float(proba[i])) for i in top_indices]
        return results
