"""
第三步：Q-learning 算法
=====================================

Q-learning 是最经典的强化学习算法，大白话理解：

  Q表 = 一张大表格，记录"在某个状态下，做某个动作有多好"
        行 = 状态（我们会把连续状态分成若干个桶）
        列 = 动作（0=左，1=右）
        值 = Q值，越大说明越好

  学习过程：
    1. 做一个动作，得到奖励
    2. 根据"实际结果"更新Q表里的估计值
    3. 重复，Q表越来越准确
    4. 最终按Q表选最好动作

  核心公式（贝尔曼方程）：
    Q[s][a] ← Q[s][a] + α × (r + γ × max(Q[s']) - Q[s][a])

    其中：
      α (alpha) = 学习率，每次更新多少（0~1，类似学习速度）
      γ (gamma) = 折扣因子，未来奖励打几折（0~1，越大越看重未来）
      r          = 刚刚得到的奖励
      s'         = 新状态
      max(Q[s']) = 新状态下最好的Q值（未来期望）

  ε-greedy 策略（探索 vs 利用）：
    - 以 ε 的概率：随机探索（去试没试过的动作）
    - 以 1-ε 的概率：选Q值最大的动作（利用已学到的知识）
    - ε 随训练进行逐渐减小（越来越自信，探索越来越少）

预期结果：训练500轮后，平均得分超过150分（随机基线只有~22分）
"""

import gymnasium as gym
import numpy as np

# ── 超参数 ──────────────────────────────────────────────────
NUM_EPISODES = 1500     # 训练总轮数
ALPHA = 0.1            # 学习率：每次更新Q值的幅度
GAMMA = 0.99           # 折扣因子：未来奖励打99折
EPSILON_START = 1.0    # 初始探索率：一开始全靠随机探索
EPSILON_END = 0.01     # 最终探索率：训练后期还保留1%的随机性
EPSILON_DECAY = 0.995  # 每轮结束后 ε 乘以这个数（逐渐减小）

# 状态离散化：把连续数值分成若干个"桶"
# CartPole的状态是连续的（无限多个值），Q表需要离散化才能建表
NUM_BINS = 10          # 每个维度分10个桶
# ────────────────────────────────────────────────────────────

env = gym.make("CartPole-v1")

# 定义每个状态维度的值域范围（超出范围的截断）
STATE_BOUNDS = [
    (-2.4, 2.4),    # 小车位置
    (-3.0, 3.0),    # 小车速度（实际上无界，这里截断）
    (-0.21, 0.21),  # 杆角度（约±12度）
    (-3.0, 3.0),    # 杆角速度（实际上无界，这里截断）
]

# Q表：形状 = (10, 10, 10, 10, 2)
# 前4维对应4个状态维度各10个桶，最后1维对应2个动作
q_table = np.zeros([NUM_BINS] * 4 + [env.action_space.n])


def discretize(state):
    """把连续状态转换成离散的桶编号（tuple，用于Q表的索引）"""
    bins = []
    for i, (val, (low, high)) in enumerate(zip(state, STATE_BOUNDS)):
        val = np.clip(val, low, high)           # 超出范围的截断
        bucket = int((val - low) / (high - low) * NUM_BINS)  # 映射到桶
        bucket = min(bucket, NUM_BINS - 1)      # 防止越界
        bins.append(bucket)
    return tuple(bins)


def choose_action(state_idx, epsilon):
    """ε-greedy策略：以ε概率随机，以1-ε概率选最优"""
    if np.random.random() < epsilon:
        return env.action_space.sample()        # 随机探索
    else:
        return np.argmax(q_table[state_idx])    # 贪心利用


# ── 训练循环 ────────────────────────────────────────────────
scores = []
epsilon = EPSILON_START

print("Q-learning 训练开始...")
print(f"Q表大小：{q_table.shape}（共 {q_table.size} 个格子）\n")

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state_idx = discretize(state)
    total_reward = 0

    while True:
        # 1. 选动作
        action = choose_action(state_idx, epsilon)

        # 2. 执行动作，观察结果
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_idx = discretize(next_state)
        total_reward += reward

        # 3. 更新Q表（贝尔曼方程）
        old_q = q_table[state_idx][action]
        if done:
            # 游戏结束，没有未来奖励
            target = reward
        else:
            # 未来奖励 = 新状态下最好的Q值，乘以折扣因子
            target = reward + GAMMA * np.max(q_table[next_state_idx])

        # 向target靠近一点点（学习率控制步伐大小）
        q_table[state_idx][action] = old_q + ALPHA * (target - old_q)

        state_idx = next_state_idx

        if done:
            break

    # 每轮结束后降低探索率
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    scores.append(total_reward)

    # 每100轮打印一次进度
    if (episode + 1) % 100 == 0:
        avg = np.mean(scores[-100:])
        print(f"第 {episode + 1:4d} 轮 | 最近100轮平均分：{avg:.1f} | ε={epsilon:.3f}")

env.close()
# ────────────────────────────────────────────────────────────

print("\n训练完成！")
print(f"最后100轮平均得分：{np.mean(scores[-100:]):.1f}（满分200）")
print(f"（随机基线约22分，提升了 {np.mean(scores[-100:]) / 22:.1f}x）")

# 保存结果
np.save("q_learning_scores.npy", scores)
np.save("q_table.npy", q_table)
print("\n（分数和Q表已保存，运行 plot_results.py 查看学习曲线）")
