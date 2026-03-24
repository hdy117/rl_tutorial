#!/usr/bin/env python3
"""
Policy Gradient 可视化图表生成脚本
生成：方差对比图、PG 技术演进树、G_t vs Advantage 时间轴分解图
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体（如果需要中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

output_dir = '/media/dhu/dhu/work/git/rl_tutorial/chapters/plots'


def generate_variance_comparison():
    """图 1: REINFORCE vs Actor-Critic 方差对比"""
    
    np.random.seed(42)
    
    # 模拟训练数据 (REINFORCE 高方差，AC 低方差)
    episodes = np.arange(0, 1001, 50)
    
    # REINFORCE: 波动大，收敛慢
    reinforce_rewards = []
    for ep in episodes:
        base_reward = min(ep * 0.6, 500)
        noise = np.random.normal(0, 80)  # 大方差
        reinforce_rewards.append(max(0, base_reward + noise))
    
    # Actor-Critic: 波动小，收敛快
    ac_rewards = []
    for ep in episodes:
        base_reward = min(ep * 1.5, 500)
        noise = np.random.normal(0, 25)  # 小方差
        ac_rewards.append(max(0, base_reward + noise))
    
    # 绘制图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：训练曲线对比
    ax1.plot(episodes, reinforce_rewards, 'r-', linewidth=1.5, label='REINFORCE', alpha=0.7)
    ax1.plot(episodes, ac_rewards, 'b-', linewidth=2, label='Actor-Critic', alpha=0.9)
    ax1.axhline(y=500, color='g', linestyle='--', linewidth=1, label='目标 (500 分)')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('训练曲线对比：REINFORCE vs Actor-Critic', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 右图：方差分布直方图
    reinforce_last_20 = reinforce_rewards[-20:]
    ac_last_20 = ac_rewards[-20:]
    
    ax2.hist(reinforce_last_20, bins=10, alpha=0.6, label='REINFORCE', color='red')
    ax2.hist(ac_last_20, bins=10, alpha=0.6, label='Actor-Critic', color='blue')
    ax2.set_xlabel('Reward Range', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('最后 20 次 episode 的奖励分布 (方差对比)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'variance_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存：{filepath}")
    plt.close()


def generate_pg_family_tree():
    """图 2: Policy Gradient 技术演进树（ASCII + 彩色版本）"""
    
    # ASCII 版本
    ascii_tree = """
Policy Gradient (直接优化策略)
│
├─ REINFORCE (Monte Carlo PG)
│  ├─ 核心：∇J ≈ G_t * ∇logπ(a|s)
│  └─ 问题：方差极大 ❌
│
├─ Baseline Reduction
│  ├─ 方法：(G_t - b(s)) * ∇logπ
│  └─ 关键: b(s)=V(s)最优
│
├─ Actor-Critic (AC)
│  ├─ Actor: π_θ(a|s) ← 策略网络
│  ├─ Critic: V_w(s) ← 价值网络  
│  └─ Advantage: A = r + γ*V(s') - V(s) (TD error)
│
└─ PPO (Proximal Policy Optimization)
   ├─ Clip Loss: min(r*A, clip(r)*A)
   ├─ Constraint: |π_new/π_old| ∈ [0.8, 1.2]
   └─ Why: 避免单次更新破坏策略 ✅

引出下一章:SAC (Soft Actor-Critic)
→ 最大熵 RL，连续控制更优
"""
    
    # 保存 ASCII 版本
    filepath_ascii = os.path.join(output_dir, 'pg_family_tree.txt')
    with open(filepath_ascii, 'w', encoding='utf-8') as f:
        f.write(ascii_tree)
    print(f"✅ 已保存：{filepath_ascii}")
    
    # 彩色版本 (用流程图方式)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')
    
    # 手动绘制树状图（用文本）
    tree_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           POLICY GRADIENT FAMILY TREE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────┐
│     Policy Gradient (θ)             │
│  Directly optimize policy π_θ(a|s)  │
└───────────────┬─────────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    v           v           v
┌───────┐  ┌─────────┐  ┌──────────┐
│REIN- │  │ BASELINE│  │ ACTOR-   │
│FORCE │  │ REDUCTION│ │ CRITIC   │
│(高方差)│  │ (降方差)│  │ (稳定高效)│
└───────┘  └─────────┘  └────┬─────┘
    │              │          │
    v              v          v
G_t * ∇logπ   (G_t - V(s))  TD error
  ↑              ↓           A = r + γV' - V
  │         Advantage       ↓
  │                    Actor + Critic
  │                        ↓
  └───────────────────────┘
                          │
                          v
                   ┌──────────┐
                   │    PPO   │
                   │(约束更新)│
                   └────┬─────┘
                        │
                  Clip Loss + ε
                        │
                        v
              引出下一章：SAC!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    ax.text(0.05, 0.95, tree_text, fontsize=10, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    filepath_tree = os.path.join(output_dir, 'pg_family_tree.png')
    plt.savefig(filepath_tree, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存：{filepath_tree}")
    plt.close()


def generate_advantage_vs_return():
    """图 3: G_t vs Advantage 时间轴分解"""
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    
    # 模拟数据：CartPole episode
    timesteps = list(range(0, 51))
    rewards = np.random.randn(50) * 10 + 5  # 随机奖励
    discounts = [0.99**i for i in range(50)]
    
    # 计算 G_t (从 t 时刻起的折扣回报)
    G_t = []
    cumsum = 0
    for r, d in zip(reversed(rewards), reversed(discounts)):
        cumsum = r + d * cumsum
        G_t.insert(0, cumsum)
    
    # 计算 V(s) (假设 Critic 预测的期望价值)
    V_s = np.ones_like(G_t) * 50  # 平均预期
    
    # Advantage = G_t - V(s)
    A_st_a = [g - v for g, v in zip(G_t, V_s)]
    
    # 图 1: G_t 时间轴
    axs[0].plot(timesteps, G_t, 'r-', linewidth=2, label='G_t (总回报)', marker='o', markersize=3)
    axs[0].axhline(y=50, color='g', linestyle='--', label='V(s) = 50')
    axs[0].set_xlabel('Time Step t', fontsize=12)
    axs[0].set_ylabel('Return G_t', fontsize=12)
    axs[0].set_title('图 A: G_t (总回报) - 包含与当前动作无关的奖励', fontsize=14, fontweight='bold')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # 高亮几个时间步
    highlight_steps = [5, 20, 35]
    for step in highlight_steps:
        axs[0].axvline(x=step, color='orange', linestyle=':', alpha=0.5)
        axs[0].annotate(f't={step}', xy=(step, G_t[step]), 
                       xytext=(step+2, G_t[step]+15),
                       fontsize=9, arrowprops=dict(arrowstyle='->'))
    
    # 图 2: Advantage 时间轴
    axs[1].plot(timesteps, A_st_a, 'b-', linewidth=2, label='A(s,a) = G_t - V(s)', marker='s', markersize=3)
    axs[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axs[1].set_xlabel('Time Step t', fontsize=12)
    axs[1].set_ylabel('Advantage A(s,a)', fontsize=12)
    axs[1].set_title('图 B: Advantage - 相对平均表现好多少？(方差更小！)', fontsize=14, fontweight='bold')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    
    # 标注关键信息
    axs[1].annotate('A > 0: 好于平均 → ↑概率', xy=(25, 30), xytext=(15, 45),
                   fontsize=10, arrowprops=dict(arrowstyle='->'), color='green')
    axs[1].annotate('A < 0: 差于平均 → ↓概率', xy=(25, -30), xytext=(15, -45),
                   fontsize=10, arrowprops=dict(arrowstyle='->'), color='red')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'advantage_vs_return.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存：{filepath}")
    plt.close()


def generate_gaussian_evolution():
    """图 4: 高斯策略 μ, σ演化动画的关键帧"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    timesteps = [0, 250, 500]  # t=0, t=250, t=500
    
    for ax, t in zip(axes, timesteps):
        x = np.linspace(-3, 3, 100)
        
        if t == 0:
            mu, std = 0.0, 1.5  # 初始化：宽泛探索
        elif t == 250:
            mu, std = -0.3, 0.6  # 开始收敛
        else:
            mu, std = -0.5, 0.3  # 接近最优
            
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
        
        ax.plot(x, y, 'b-', linewidth=2, label=f't={t}')
        ax.axvline(x=mu, color='r', linestyle='--', label=f'mu = {mu:.1f}')
        ax.set_xlabel('Action a', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'策略分布演化 (t={t})\nμ={mu:.2f}, σ={std:.2f}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 标注探索 vs 利用
        if t == 0:
            ax.annotate('🔥 高探索\n(σ大)', xy=(mu, y.max()*0.8), 
                       fontsize=10, ha='center', color='orange')
        elif t == 250:
            ax.annotate('⚖️ 平衡\n(σ中等)', xy=(mu, y.max()*0.7),
                       fontsize=10, ha='center', color='yellow')
        else:
            ax.annotate('✅ 高利用\n(σ小，专注最优)', xy=(mu, y.max()*0.6),
                       fontsize=10, ha='center', color='green')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'gaussian_evolution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存：{filepath}")
    plt.close()


def main():
    """生成所有图表"""
    print("=" * 60)
    print("🎨 Policy Gradient 可视化图表生成")
    print("=" * 60)
    
    generate_variance_comparison()
    generate_pg_family_tree()
    generate_advantage_vs_return()
    generate_gaussian_evolution()
    
    print("=" * 60)
    print("✅ 所有图表已生成完毕！")
    print(f"📁 输出目录：{output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
