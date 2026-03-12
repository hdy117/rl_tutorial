"""
画训练曲线
=====================================

运行此脚本前请先运行：
  python 02_random_agent.py
  python 03_q_learning.py

会生成 training_curve.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('强化学习实验结果：CartPole-v1', fontsize=14, fontweight='bold')

# ── 左图：Q-learning 学习曲线 ────────────────────────────────
ax1 = axes[0]
if os.path.exists("q_learning_scores.npy"):
    ql_scores = np.load("q_learning_scores.npy")
    window = 20
    smoothed = np.convolve(ql_scores, np.ones(window) / window, mode='valid')

    ax1.plot(ql_scores, alpha=0.3, color='steelblue', label='每轮得分')
    ax1.plot(range(window - 1, len(ql_scores)), smoothed,
             color='steelblue', linewidth=2, label=f'{window}轮平均')
    ax1.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='满分(200)')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('得分')
    ax1.set_title('Q-learning 学习过程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 220)
else:
    ax1.text(0.5, 0.5, '请先运行\npython 03_q_learning.py',
             ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.set_title('Q-learning 学习过程')

# ── 右图：随机 vs Q-learning 对比 ────────────────────────────
ax2 = axes[1]
has_random = os.path.exists("random_scores.npy")
has_ql = os.path.exists("q_learning_scores.npy")

labels, means, stds, colors = [], [], [], []

if has_random:
    r = np.load("random_scores.npy")
    labels.append('随机策略')
    means.append(np.mean(r))
    stds.append(np.std(r))
    colors.append('salmon')

if has_ql:
    ql = np.load("q_learning_scores.npy")
    labels.append('Q-learning\n(后100轮)')
    means.append(np.mean(ql[-100:]))
    stds.append(np.std(ql[-100:]))
    colors.append('steelblue')

if labels:
    bars = ax2.bar(labels, means, yerr=stds, capsize=8,
                   color=colors, alpha=0.8, edgecolor='black')
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    ax2.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='满分(200)')
    ax2.set_ylabel('平均得分')
    ax2.set_title('随机 vs 学习后的对比')
    ax2.set_ylim(0, 230)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
else:
    ax2.text(0.5, 0.5, '请先运行\n02_random_agent.py\n03_q_learning.py',
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('随机 vs 学习后的对比')

plt.tight_layout()
plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
print("图表已保存到 training_curve.png")
plt.show()
