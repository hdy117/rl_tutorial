"""
第二步：随机智能体（基线）
=====================================

随机策略 = 每次随机选动作，不管当前状态是什么

用途：
  - 建立"最差情况"基线
  - 对比后面学习算法的提升

预期结果：平均得分约 20~25 分（满分200）
"""

import gymnasium as gym
import numpy as np

# env = gym.make("CartPole-v1")
env = gym.make("CartPole-v1", sutton_barto_reward=True)

NUM_EPISODES = 200  # 玩200轮
scores = []

print("随机智能体正在游戏中...")
print("（每次随机选左或右，不用任何学习）\n")

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    step_counter = 0

    while True:
        # 随机选动作：从动作空间里随机取一个
        action = env.action_space.sample()  # 等价于 random.choice([0, 1])
        step_counter += 1

        next_state, reward, terminated, truncated, _ = env.step(action)
        # total_reward += reward

        if terminated or truncated:
            total_reward += -1
            break
        else:
            total_reward+=1

        state = next_state
    print(f"第 {episode + 1} 轮结束，得分：{total_reward}，步数：{step_counter}")
    scores.append(total_reward)

env.close()

# 统计结果
print(f"结果统计（共 {NUM_EPISODES} 轮）：")
print(f"  平均得分：{np.mean(scores):.1f}")
print(f"  最高得分：{np.max(scores):.0f}")
print(f"  最低得分：{np.min(scores):.0f}")
print(f"  标准差  ：{np.std(scores):.1f}")

print("\n结论：随机乱猜只能得约20分，明显不够好。")
print("下一步：运行 03_q_learning.py 用Q-learning让智能体真正学习！")

# 保存分数供后续对比
np.save("random_scores.npy", scores)
print("\n（分数已保存到 random_scores.npy）")
