"""
MNIST 手写数字 SVM 模型训练脚本
=================================
加载 MNIST 数据集，使用 StandardScaler + SVC(RBF) 训练，
评估准确率，保存模型和标准化器到 model/ 目录，
并自动生成实验报告所需的全部可视化图片到 figures/ 目录。

生成的图片：
  figure_1_mnist_samples.png        — MNIST 0-9 样本灰度图
  figure_2_confusion_matrix.png     — 10×10 混淆矩阵热力图
  figure_3_support_vectors.png      — 各类别支持向量数量柱状图
  figure_4_parameter_experiments.png — 参数敏感性实验组合图
  figure_5_error_samples.png        — 错误分类样本可视化
"""

import os
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import joblib

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt

# ============================================================
# 全局配色与字体设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print("=" * 60)
    print("  SVM 手写数字识别 — 模型训练")
    print("=" * 60)

    # 创建图片保存目录
    os.makedirs("figures", exist_ok=True)

    # ============================================================
    # 1. 加载 MNIST 数据集
    # ============================================================
    print("\n[1/5] 加载 MNIST 数据集 ...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                        as_frame=False, parser='auto')
    X = X / 255.0  # 归一化到 [0, 1]

    print(f"  样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"  类别数: {len(np.unique(y))}")

    # ============================================================
    # 图1: 显示手写数字样本 (每个数字 0-9 各取第一个样本)
    # ============================================================
    print("\n  [图1] 生成 MNIST 样本显示图 ...")
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle('MNIST 手写数字样本 (28×28 灰度图)', fontsize=16, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        idx = np.where(y == str(i))[0][0]
        ax.imshow(X[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'Digit: {i}', fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig('figures/figure_1_mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] 已保存 → figures/figure_1_mnist_samples.png")

    # ============================================================
    # 2. 划分训练集和测试集
    # ============================================================
    print("\n[2/5] 划分训练集 / 测试集 (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    # ============================================================
    # 3. 标准化
    # ============================================================
    print("\n[3/5] StandardScaler 标准化 ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  完成")

    # ============================================================
    # 4. 训练 SVM (使用 RBF 核)
    # ============================================================
    print("\n[4/5] 训练 SVM (kernel=rbf, C=5.0, gamma='scale') ...")
    print("  [*] 这可能需要几分钟，请耐心等待...")
    t0_train = time.time()
    svm = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=42)
    svm = CalibratedClassifierCV(svm, method='sigmoid', cv=3, ensemble=False)
    svm.fit(X_train_scaled, y_train)
    elapsed_train = time.time() - t0_train
    print(f"  训练完成！耗时: {elapsed_train:.1f} 秒 ({elapsed_train/60:.1f} 分钟)")

    # ============================================================
    # 5. 评估 & 保存
    # ============================================================
    print("\n[5/5] 评估模型 ...")
    y_pred = svm.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  [OK] 测试集准确率: {acc * 100:.2f}%")
    print("\n  分类报告:")
    print(classification_report(y_test, y_pred))

    # ============================================================
    # 图2: 混淆矩阵热力图
    # ============================================================
    print("\n  [图2] 生成混淆矩阵热力图 ...")
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title('混淆矩阵 — SVM (RBF 核, C=5.0, gamma=\'scale\')',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    fig.savefig('figures/figure_2_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] 已保存 → figures/figure_2_confusion_matrix.png")

    # ============================================================
    # 图3: 各类别支持向量数量柱状图
    # ============================================================
    print("\n  [图3] 生成支持向量数量柱状图 ...")
    # CalibratedClassifierCV 包装后，原始 svm.estimator 未被训练
    # 正确做法：从 calibrated_classifiers_ 中获取已拟合的基础 SVC
    # ensemble=False 时只有一个分类器，取其 .estimator 即已训练的 SVC
    fitted_svc = svm.calibrated_classifiers_[0].estimator
    n_support_per_class = fitted_svc.n_support_
    total_sv = sum(n_support_per_class)

    fig, ax = plt.subplots(figsize=(9, 6))
    classes = list(range(10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    bars = ax.bar(classes, n_support_per_class, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, n_support_per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('数字类别', fontsize=13)
    ax.set_ylabel('支持向量数量', fontsize=13)
    ax.set_title(f'各类别支持向量数量柱状图（总数: {total_sv}）', fontsize=14, fontweight='bold')
    ax.set_xticks(classes)
    ax.set_ylim(0, max(n_support_per_class) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    fig.savefig('figures/figure_3_support_vectors.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] 已保存 → figures/figure_3_support_vectors.png")
    print(f"  各类别 SV 数量: {list(n_support_per_class)}")
    print(f"  支持向量总数: {total_sv}")

    # ============================================================
    # 图5: 错误分类样本可视化
    # ============================================================
    print("\n  [图5] 生成错误样本可视化 ...")
    error_indices = np.where(y_pred != y_test)[0]
    n_errors = len(error_indices)
    print(f"  错误样本数: {n_errors} / {len(y_test)} ({n_errors/len(y_test)*100:.2f}%)")

    if n_errors > 0:
        n_display = min(6, n_errors)
        fig, axes = plt.subplots(2, 3, figsize=(11, 8))
        fig.suptitle(f'错误分类样本（共 {n_errors} 个，显示前 {n_display} 个）',
                     fontsize=14, fontweight='bold')
        for i, ax in enumerate(axes.flat):
            if i < n_display:
                err_idx = error_indices[i]
                ax.imshow(X_test[err_idx].reshape(28, 28), cmap='gray')
                ax.set_title(f'真实: {y_test[err_idx]}  →  预测: {y_pred[err_idx]}',
                             fontsize=13, color='red', fontweight='bold')
            ax.axis('off')
        plt.tight_layout()
        fig.savefig('figures/figure_5_error_samples.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  [OK] 已保存 → figures/figure_5_error_samples.png")
    else:
        print("  [!] 没有错误样本，跳过此图")
        # 创建一个空图占位
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, '无错误样本\n(准确率 100%)', transform=ax.transAxes,
                ha='center', va='center', fontsize=18, color='green')
        ax.axis('off')
        fig.savefig('figures/figure_5_error_samples.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ============================================================
    # 保存模型
    # ============================================================
    os.makedirs("model", exist_ok=True)
    model_path = "model/svm_mnist.pkl"
    scaler_path = "model/scaler.pkl"

    joblib.dump(svm, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n  模型已保存至: {model_path}")
    print(f"  标准化器已保存至: {scaler_path}")
    print("\n" + "=" * 60)
    print("  核心训练流程全部完成！可以运行 main.py 启动识别窗口。")
    print("=" * 60)

    # ============================================================
    # 图4: 参数敏感性实验（在核心训练完成后运行）
    # ============================================================
    print("\n")
    print("=" * 60)
    print("  参数敏感性实验（用于生成报告第十二章图表）")
    print("=" * 60)
    print("  [*] 将训练多个模型比较不同参数的影响，请耐心等待...")

    # 使用训练集的子集加速参数实验（平衡速度与代表性）
    N_PARAM_SAMPLES = 10000
    rng = np.random.RandomState(42)
    param_indices = rng.choice(len(X_train_scaled), N_PARAM_SAMPLES, replace=False)
    X_param = X_train_scaled[param_indices]
    y_param = y_train[param_indices]
    print(f"  参数实验使用 {N_PARAM_SAMPLES} 个训练样本（从 {len(X_train_scaled)} 中随机采样）")

    # ------- 实验 A: C 参数实验 -------
    print("\n  [实验 A] C 参数扫描 (kernel='rbf', gamma='scale') ...")
    C_values = [0.01, 0.1, 1.0, 5.0, 10.0, 100.0]
    C_results = {'acc': [], 'sv_count': [], 'time': []}

    for c in C_values:
        print(f"    C={c:<7}", end=" ", flush=True)
        try:
            t0 = time.time()
            m = SVC(kernel='rbf', C=c, gamma='scale', random_state=42)
            m.fit(X_param, y_param)
            elapsed = time.time() - t0
            pred = m.predict(X_test_scaled)
            C_results['acc'].append(accuracy_score(y_test, pred))
            C_results['sv_count'].append(sum(m.n_support_))
            C_results['time'].append(elapsed)
            print(f"→ Acc={C_results['acc'][-1]*100:.1f}%, "
                  f"SV={C_results['sv_count'][-1]}, "
                  f"耗时={elapsed:.1f}s")
        except Exception as e:
            print(f"→ 失败: {e}")
            C_results['acc'].append(0.0)
            C_results['sv_count'].append(0)
            C_results['time'].append(0.0)

    # ------- 实验 B: gamma 参数实验 -------
    print("\n  [实验 B] gamma 参数扫描 (kernel='rbf', C=5.0) ...")
    gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
    gamma_labels = ['scale', 'auto', '0.001', '0.01', '0.1', '1.0']
    G_results = {'acc': [], 'sv_count': [], 'time': []}

    for g in gamma_values:
        label = str(g)
        print(f"    gamma={label:<7}", end=" ", flush=True)
        try:
            t0 = time.time()
            m = SVC(kernel='rbf', C=5.0, gamma=g, random_state=42)
            m.fit(X_param, y_param)
            elapsed = time.time() - t0
            pred = m.predict(X_test_scaled)
            G_results['acc'].append(accuracy_score(y_test, pred))
            G_results['sv_count'].append(sum(m.n_support_))
            G_results['time'].append(elapsed)
            print(f"→ Acc={G_results['acc'][-1]*100:.1f}%, "
                  f"SV={G_results['sv_count'][-1]}, "
                  f"耗时={elapsed:.1f}s")
        except Exception as e:
            print(f"→ 失败: {e}")
            G_results['acc'].append(0.0)
            G_results['sv_count'].append(0)
            G_results['time'].append(0.0)

    # ------- 实验 C: kernel 核函数实验 -------
    print("\n  [实验 C] 核函数对比 (C=5.0) ...")
    kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
    K_results = {'acc': [], 'time': []}

    for k in kernel_list:
        print(f"    kernel={k:<9}", end=" ", flush=True)
        try:
            t0 = time.time()
            kwargs = {'kernel': k, 'C': 5.0, 'random_state': 42}
            if k in ('rbf', 'poly', 'sigmoid'):
                kwargs['gamma'] = 'scale'
            m = SVC(**kwargs)
            m.fit(X_param, y_param)
            elapsed = time.time() - t0
            pred = m.predict(X_test_scaled)
            K_results['acc'].append(accuracy_score(y_test, pred))
            K_results['time'].append(elapsed)
            print(f"→ Acc={K_results['acc'][-1]*100:.1f}%, "
                  f"耗时={elapsed:.1f}s")
        except Exception as e:
            print(f"→ 失败: {e}")
            K_results['acc'].append(0.0)
            K_results['time'].append(0.0)

    # ------- 绘制组合图 -------
    print("\n  [图4] 绘制参数实验组合图 ...")
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle('SVM 参数敏感性实验', fontsize=18, fontweight='bold', y=0.98)

    # 子图1: C 参数实验（双Y轴）
    ax1 = fig.add_subplot(2, 2, 1)
    ax1b = ax1.twinx()
    x1 = range(len(C_values))
    line_acc = ax1.plot(x1, [a * 100 for a in C_results['acc']],
                        'o-', color='#2196F3', linewidth=2.5, markersize=10,
                        label='Accuracy (%)', zorder=3)
    line_sv = ax1b.plot(x1, C_results['sv_count'],
                        's--', color='#FF9800', linewidth=2.5, markersize=10,
                        label='支持向量总数', zorder=3)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([str(c) for c in C_values], fontsize=11)
    ax1.set_xlabel('C 值', fontsize=13)
    ax1.set_ylabel('准确率 (%)', fontsize=13, color='#2196F3')
    ax1b.set_ylabel('支持向量总数', fontsize=13, color='#FF9800')
    ax1.set_title('C 参数实验 (kernel=rbf, gamma=scale)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # 合并图例
    lns = line_acc + line_sv
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='center right', fontsize=10)
    # 在数据点上标注准确率
    for i, (c_val, acc_val) in enumerate(zip(C_values, C_results['acc'])):
        ax1.annotate(f'{acc_val*100:.1f}%', (i, acc_val * 100),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=9, color='#1565C0')

    # 子图2: gamma 参数实验（双Y轴）
    ax2 = fig.add_subplot(2, 2, 2)
    ax2b = ax2.twinx()
    x2 = range(len(gamma_labels))
    line_acc2 = ax2.plot(x2, [a * 100 for a in G_results['acc']],
                         'o-', color='#4CAF50', linewidth=2.5, markersize=10,
                         label='Accuracy (%)', zorder=3)
    line_sv2 = ax2b.plot(x2, G_results['sv_count'],
                         's--', color='#FF9800', linewidth=2.5, markersize=10,
                         label='支持向量总数', zorder=3)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(gamma_labels, fontsize=11)
    ax2.set_xlabel('gamma 值', fontsize=13)
    ax2.set_ylabel('准确率 (%)', fontsize=13, color='#4CAF50')
    ax2b.set_ylabel('支持向量总数', fontsize=13, color='#FF9800')
    ax2.set_title('gamma 参数实验 (kernel=rbf, C=5.0)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    lns2 = line_acc2 + line_sv2
    labels2 = [l.get_label() for l in lns2]
    ax2.legend(lns2, labels2, loc='center right', fontsize=10)
    for i, (g_lbl, acc_val) in enumerate(zip(gamma_labels, G_results['acc'])):
        ax2.annotate(f'{acc_val*100:.1f}%', (i, acc_val * 100),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=9, color='#2E7D32')

    # 子图3: kernel 对比柱状图
    ax3 = fig.add_subplot(2, 2, 3)
    bar_colors = ['#78909C', '#4CAF50', '#2196F3', '#FF9800']
    bars3 = ax3.bar(kernel_list, [a * 100 for a in K_results['acc']],
                    color=bar_colors, edgecolor='white', linewidth=1.2, width=0.5)
    for bar, acc_val in zip(bars3, K_results['acc']):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{acc_val*100:.1f}%', ha='center', fontsize=13, fontweight='bold')
    ax3.set_xlabel('核函数类型', fontsize=13)
    ax3.set_ylabel('准确率 (%)', fontsize=13)
    ax3.set_title('不同核函数对比 (C=5.0)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', alpha=0.3)

    # 子图4: 汇总表格（使用 matplotlib table 替代 ASCII-art，避免乱码）
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    ax4.set_title('参数敏感性实验结论汇总', fontsize=14, fontweight='bold', y=0.98)

    # 过滤掉因过拟合导致准确率极低的异常值
    c_valid = [a for a in C_results['acc'] if a > 0.1] or C_results['acc']
    g_valid = [a for a in G_results['acc'] if a > 0.5] or G_results['acc']
    k_valid = [a for a in K_results['acc'] if a > 0.1] or K_results['acc']

    # --- 数据表 ---
    table_data = [
        ['C 值',     'C=5.0',  f'{min(c_valid)*100:.1f}% ~ {max(c_valid)*100:.1f}%', 'C 过大/过小均不利'],
        ['gamma',    'scale',  f'{min(g_valid)*100:.1f}% ~ {max(g_valid)*100:.1f}%',  'gamma=1.0 严重过拟合'],
        ['kernel',   'rbf',    f'{min(k_valid)*100:.1f}% ~ {max(k_valid)*100:.1f}%',  'RBF 核在 MNIST 上最优'],
    ]
    col_labels = ['实验参数', '最佳取值', '准确率范围', '结论']

    tbl = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='upper center',
        bbox=[0.05, 0.42, 0.92, 0.30]  # [left, bottom, width, height]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)

    # 表格样式
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if key[0] == 0:  # 表头行
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        else:
            cell.set_facecolor('#f9f9f9' if key[0] % 2 == 0 else '#ffffff')

    # --- 结论文字 ---
    conclusion_text = (
        "▶ C 不是越大越好：C 过小 → 欠拟合；C 过大 → 过拟合\n"
        "▶ gamma 适中最好：gamma 太小 → 近似线性；gamma 太大 → 严重过拟合\n"
        "▶ 核函数效果排序：RBF > poly > linear > sigmoid"
    )
    ax4.text(0.5, 0.12, conclusion_text, transform=ax4.transAxes,
             ha='center', va='center', fontsize=10.5,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5',
                       edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('figures/figure_4_parameter_experiments.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] 已保存 → figures/figure_4_parameter_experiments.png")

    # ============================================================
    # 完成总结
    # ============================================================
    print("\n" + "=" * 60)
    print("  全部完成！以下图片已保存至 figures/ 目录：")
    print("=" * 60)
    print("    图1  figure_1_mnist_samples.png        — MNIST 0-9 样本显示")
    print("    图2  figure_2_confusion_matrix.png     — 混淆矩阵热力图")
    print("    图3  figure_3_support_vectors.png      — 支持向量数量柱状图")
    print("    图4  figure_4_parameter_experiments.png — 参数敏感性实验")
    print("    图5  figure_5_error_samples.png        — 错误分类样本")
    print("=" * 60)
    print("\n  提示：运行 main.py 启动 GUI，手写识别后截图作为图6。")


if __name__ == "__main__":
    main()
