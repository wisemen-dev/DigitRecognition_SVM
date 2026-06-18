"""
SVM 手写数字识别 — PyQt5 图形界面
==================================
提供美观的手写数字识别窗口：
- 280×280 深色绘制画布（鼠标手写）
- 实时预测结果显示（大字数字 + 置信度条）
- 识别 / 清除按钮
- 深色科技风主题
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFrame, QApplication
)
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QPen, QPixmap, QColor, QFont, QFontDatabase,
    QLinearGradient, QBrush, QPalette
)
import sys


# ============================================================
# 全局样式表 (QSS) — 深色科技风主题
# ============================================================

STYLE_QSS = """
/* === 全局 === */
* {
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
}

QMainWindow {
    background-color: #0f0f1a;
}

QWidget#centralWidget {
    background-color: #0f0f1a;
}

/* === 标题 === */
QLabel#titleLabel {
    color: #ffffff;
    font-size: 30px;
    font-weight: bold;
    padding: 16px 0px 10px 0px;
}

QLabel#subtitleLabel {
    color: #8888aa;
    font-size: 15px;
    padding-bottom: 12px;
}

/* === 卡片容器 === */
QFrame#canvasCard, QFrame#resultCard {
    background-color: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 14px;
}

QFrame#canvasCard {
    min-width: 600px;
    min-height: 600px;
}

QFrame#resultCard {
    min-width: 400px;
    min-height: 600px;
}

/* === 结果标签 === */
QLabel#resultDigit {
    color: #00ff88;
    font-size: 160px;
    font-weight: bold;
}

QLabel#resultLabel {
    color: #aaaacc;
    font-size: 18px;
}

QLabel#confidenceLabel {
    color: #ccccee;
    font-size: 18px;
}

/* === 进度条 === */
QProgressBar {
    background-color: #2a2a4a;
    border: none;
    border-radius: 12px;
    height: 22px;
    text-align: center;
    font-size: 12px;
    color: #ffffff;
}

QProgressBar::chunk {
    border-radius: 12px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00cc66, stop:1 #00ff88);
}

/* === 按钮 === */
QPushButton {
    border: none;
    border-radius: 14px;
    padding: 14px 44px;
    font-size: 20px;
    font-weight: bold;
    color: #ffffff;
    min-width: 140px;
}

QPushButton:hover {
    transform: translateY(-1px);
}

QPushButton:pressed {
    transform: translateY(1px);
}

QPushButton#predictBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00b359, stop:1 #00e673);
}

QPushButton#predictBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00cc66, stop:1 #00ff80);
}

QPushButton#predictBtn:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00994d, stop:1 #00cc66);
}

QPushButton#clearBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e0556a, stop:1 #ff6b7f);
}

QPushButton#clearBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e8687b, stop:1 #ff808f);
}

QPushButton#clearBtn:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #c04050, stop:1 #e0556a);
}

/* === 状态栏 === */
QLabel#statusLabel {
    color: #666688;
    font-size: 14px;
    padding-top: 8px;
}
"""


# ============================================================
# 绘制画布
# ============================================================

class DrawCanvas(QLabel):
    """560×560 手写画布，支持鼠标绘制白色笔画。"""

    drawing_changed = pyqtSignal()

    CANVAS_SIZE = 560
    PEN_WIDTH = 28

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.CANVAS_SIZE, self.CANVAS_SIZE)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

        # 创建像素图
        self._pixmap = QPixmap(self.CANVAS_SIZE, self.CANVAS_SIZE)
        self._pixmap.fill(QColor(0, 0, 0))
        self.setPixmap(self._pixmap)

        self._last_point: QPoint | None = None
        self._drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self._last_point = event.pos()
            self._draw_point(event.pos())

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._draw_line(self._last_point, event.pos())
            self._last_point = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = False
            self._last_point = None
            self.drawing_changed.emit()

    def _draw_point(self, pos: QPoint):
        painter = QPainter(self._pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(255, 255, 255), self.PEN_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPoint(pos)
        painter.end()
        self.setPixmap(self._pixmap)

    def _draw_line(self, from_pos: QPoint, to_pos: QPoint):
        painter = QPainter(self._pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(255, 255, 255), self.PEN_WIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(from_pos, to_pos)
        painter.end()
        self.setPixmap(self._pixmap)

    def clear(self):
        """清空画布。"""
        self._pixmap.fill(QColor(0, 0, 0))
        self.setPixmap(self._pixmap)
        self.drawing_changed.emit()

    def get_image(self):
        """返回当前画布的 QPixmap 副本。"""
        return self._pixmap.copy()

    def is_blank(self) -> bool:
        """检查画布是否为空（全黑）。"""
        image = self._pixmap.toImage()
        for y in range(self.CANVAS_SIZE):
            for x in range(self.CANVAS_SIZE):
                if image.pixelColor(x, y).value() > 10:
                    return False
        return True


# ============================================================
# 主窗口
# ============================================================

class RecognitionWindow(QMainWindow):
    """手写数字识别主窗口。"""

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self._init_ui()
        self._auto_predict_timer = QTimer()
        self._auto_predict_timer.setSingleShot(True)
        self._auto_predict_timer.setInterval(600)  # 600ms 防抖
        self._auto_predict_timer.timeout.connect(self._on_predict)

    def _init_ui(self):
        self.setWindowTitle("手写数字识别系统 — SVM")
        self.setFixedSize(1200, 880)
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        # 居中显示
        screen = QApplication.primaryScreen().availableGeometry()
        x = (screen.width() - 1200) // 2
        y = (screen.height() - 880) // 2
        self.move(x, y)

        # --- 中央组件 ---
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(30, 20, 30, 24)
        root_layout.setSpacing(12)

        # --- 标题 ---
        title = QLabel("✍️  手写数字识别系统")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)

        subtitle = QLabel("基于 SVM (支持向量机) · MNIST 数据集 · RBF 核")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(subtitle)

        # --- 主体水平布局：画布 | 结果 ---
        body_layout = QHBoxLayout()
        body_layout.setSpacing(20)

        # 左侧：画布卡片
        canvas_card = QFrame()
        canvas_card.setObjectName("canvasCard")
        canvas_layout = QVBoxLayout(canvas_card)
        canvas_layout.setContentsMargins(18, 18, 18, 18)
        canvas_layout.setAlignment(Qt.AlignCenter)

        self.canvas = DrawCanvas()
        self.canvas.drawing_changed.connect(self._on_drawing_changed)
        canvas_layout.addWidget(self.canvas)

        body_layout.addWidget(canvas_card)

        # 右侧：结果卡片
        result_card = QFrame()
        result_card.setObjectName("resultCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(20, 24, 20, 24)
        result_layout.setAlignment(Qt.AlignCenter)
        result_layout.setSpacing(10)

        result_label = QLabel("预测结果")
        result_label.setObjectName("resultLabel")
        result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(result_label)

        self.result_digit = QLabel("—")
        self.result_digit.setObjectName("resultDigit")
        self.result_digit.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_digit)

        self.confidence_label = QLabel("置信度: —")
        self.confidence_label.setObjectName("confidenceLabel")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.confidence_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(320)
        result_layout.addWidget(self.progress_bar, alignment=Qt.AlignCenter)

        # Top-K 备选结果显示
        self.topk_label = QLabel("")
        self.topk_label.setObjectName("topkLabel")
        self.topk_label.setStyleSheet("color: #7777aa; font-size: 16px; padding-top: 12px;")
        self.topk_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.topk_label)

        result_layout.addStretch()

        body_layout.addWidget(result_card)
        root_layout.addLayout(body_layout)

        # --- 按钮栏 ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.setAlignment(Qt.AlignCenter)

        self.predict_btn = QPushButton("🔍  识  别")
        self.predict_btn.setObjectName("predictBtn")
        self.predict_btn.setCursor(Qt.PointingHandCursor)
        self.predict_btn.clicked.connect(self._on_predict)
        btn_layout.addWidget(self.predict_btn)

        self.clear_btn = QPushButton("🗑️  清  除")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self.clear_btn)

        root_layout.addLayout(btn_layout)

        # --- 状态栏 ---
        self.status_label = QLabel("🖱️  在左侧画布上用鼠标绘制数字，松开鼠标后自动识别")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(self.status_label)

        # 应用全局样式
        self.setStyleSheet(STYLE_QSS)

    # --- 槽函数 ---

    def _on_drawing_changed(self):
        """画布内容变化时自动触发防抖预测。"""
        if not self.canvas.is_blank():
            self._auto_predict_timer.start()
        else:
            self._reset_result()

    def _on_predict(self):
        """执行识别。"""
        if self.canvas.is_blank():
            self.status_label.setText("⚠️  请先在画布上绘制数字！")
            self.status_label.setStyleSheet("color: #ff6b7f; font-size: 15px; padding-top: 8px;")
            return

        pixmap = self.canvas.get_image()
        image = pixmap.toImage()

        # 转换为 PIL Image 用的原始数据
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        from PIL import Image
        pil_image = Image.frombytes(
            "RGBA", (image.width(), image.height()),
            bytes(ptr), "raw", "BGRA", 0, 1
        )

        # 预测
        digit, confidence = self.recognizer.predict(pil_image)

        # 更新 UI
        self.result_digit.setText(str(digit))
        self.confidence_label.setText(f"置信度: {confidence * 100:.1f}%")
        self.progress_bar.setValue(int(confidence * 100))

        # 根据置信度调整进度条颜色
        if confidence > 0.9:
            color = "#00ff88"
        elif confidence > 0.7:
            color = "#ffcc00"
        else:
            color = "#ff6b7f"

        # Top-K
        try:
            top_k = self.recognizer.predict_top_k(pil_image, k=3)
            topk_text = " · ".join([f"{d} ({p*100:.1f}%)" for d, p in top_k])
            self.topk_label.setText(f"Top-3:  {topk_text}")
        except Exception:
            self.topk_label.setText("")

        self.status_label.setText(f"✅  识别完成！预测数字为 {digit}")
        self.status_label.setStyleSheet(
            f"color: {color}; font-size: 15px; padding-top: 8px;"
        )

    def _on_clear(self):
        """清空画布和结果。"""
        self.canvas.clear()
        self._reset_result()
        self.status_label.setText("🖱️  在左侧画布上用鼠标绘制数字，松开鼠标后自动识别")
        self.status_label.setStyleSheet("color: #666688; font-size: 15px; padding-top: 8px;")

    def _reset_result(self):
        """重置结果显示。"""
        self.result_digit.setText("—")
        self.confidence_label.setText("置信度: —")
        self.progress_bar.setValue(0)
        self.topk_label.setText("")


# ============================================================
# 启动入口
# ============================================================

def launch_gui(recognizer):
    """启动 PyQt5 GUI 并传入识别器实例。"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 设置暗色调色板作为后备
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(15, 15, 26))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(26, 26, 46))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(30, 30, 50))
    app.setPalette(palette)

    window = RecognitionWindow(recognizer)
    window.show()
    sys.exit(app.exec_())
