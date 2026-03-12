"""
第一步：认识强化学习的基本概念
=====================================

强化学习 = 让"智能体"在环境里试错，学会最好的策略

这个脚本演示：
  - 什么是 state（状态）
  - 什么是 action（动作）
  - 什么是 reward（奖励）
  - 什么是 episode（一轮游戏）

环境：CartPole-v1
  场景：一根杆子立在小车上，你控制小车左右移动来保持杆不倒
  目标：杆保持竖直的时间越长越好（最多200步）
"""

import gymnasium as gym

print("=" * 50)
print("欢迎来到强化学习入门实验！")
print("=" * 50)

# 创建环境（就像打开一个游戏）
env = gym.make("CartPole-v1", sutton_barto_reward=True)

# 重置环境，开始新一轮游戏，获得初始状态
state, info = env.reset()

print("\n【环境说明】")
print(f"动作空间：{env.action_space}  → 只有2个动作：0=向左推, 1=向右推")
print(f"状态空间：{env.observation_space}  → 4个连续数值")

print("\n【状态说明】state是一个包含4个数的数组：")
print(f"  state[0] = {state[0]:.4f}  → 小车位置（负=左，正=右）")
print(f"  state[1] = {state[1]:.4f}  → 小车速度")
print(f"  state[2] = {state[2]:.4f}  → 杆的角度（负=左倾，正=右倾）")
print(f"  state[3] = {state[3]:.4f}  → 杆的角速度")

print("\n【手动运行5步，观察每步的变化】")
print("-" * 50)

for step in range(50):
    # 动作：交替选左(0)右(1)
    action = step % 2
    action_name = "向左推" if action == 0 else "向右推"

    # 执行动作，环境返回新状态、奖励、是否结束
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"\n第 {step + 1} 步：{action_name}（action={action}）")
    print(f"  新状态：位置={next_state[0]:.4f}, 角度={next_state[2]:.4f}")
    print(f"  奖励：{reward}  （杆没倒就得1分！）")
    print(f"  游戏结束？{'是' if done else '否'}")

    state = next_state
    print(f"terminated:{terminated}, truncated:{truncated}")

    if done:
        print("\n  游戏结束了！")
        break

env.close()

print("\n" + "=" * 50)
print("核心概念总结：")
print("  state  = 环境当前的样子（4个数字）")
print("  action = 我们做的决定（0或1）")
print("  reward = 做了决定后得到的反馈（+1/-1）")
print("  done   = 这轮游戏是否结束")
print("\n下一步：运行 02_random_agent.py 看随机策略的表现")
print("=" * 50)
