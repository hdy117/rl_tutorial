#!/usr/bin/env python3
"""
REINFORCE 算法完整实现
Policy Gradient 的最基础版本：用整条轨迹回报 G_t 更新策略

适用环境：离散动作空间 (CartPole, FrozenLake 等)
依赖：gymnasium, torch, numpy, matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    """策略网络：状态 -> 动作概率分布 (离散)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # logits
        )
    
    def forward(self, state):
        """返回 log_softmax，方便计算对数概率"""
        logits = self.net(state)
        return F.log_softmax(logits, dim=-1)


class REINFORCE:
    """REINFORCE 算法实现 (Monte Carlo Policy Gradient)"""
    
    def __init__(self, policy, lr=1e-2, gamma=0.99):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
    
    @staticmethod
    def compute_returns(rewards, gamma=0.99):
        """
        计算折扣回报 G_t
        
        Args:
            rewards: [r_0, r_1, ..., r_T]
            gamma: 折扣因子
        
        Returns:
            returns: [G_0, G_1, ..., G_T] (从 t 时刻起的折扣回报和)
        """
        returns = []
        G = 0
        
        # 反向遍历：G_t = r_t + γ*G_{t+1}
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        return torch.tensor(returns, dtype=torch.float32)
    
    def train_step(self, states, actions, returns):
        """
        训练一步 (一个 episode)
        
        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            returns: [batch_size] G_t 值
        
        Returns:
            loss: 标量损失值
        """
        # 计算对数概率
        log_probs = self.policy(states)  # [batch_size, action_dim]
        
        # 选择对应动作的对数概率
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(-1))  # [batch_size, 1]
        
        # Policy Gradient Loss: -E[log_prob * G_t]
        # (最大化期望回报 = 最小化负对数概率 * 回报)
        loss = -(chosen_log_probs.squeeze() * returns).mean()
        
        # 反向传播 & 更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item()


def train_reinforce(env_name="CartPole-v1", 
                   num_episodes=2000,
                   hidden_dim=128,
                   lr=1e-2,
                   gamma=0.99):
    """
    训练主循环
    
    Args:
        env_name: Gymnasium 环境名称
        num_episodes: 训练轮数
        hidden_dim: 隐藏层维度
        lr: 学习率
        gamma: 折扣因子
    """
    # 初始化环境和策略网络
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNet(state_dim, action_dim, hidden_dim)
    agent = REINFORCE(policy, lr=lr, gamma=gamma)
    
    # 训练统计
    episode_rewards = []
    running_reward = None
    
    print(f"\n🚀 开始训练 {env_name}")
    print("=" * 60)
    print(f"参数：hidden_dim={hidden_dim}, lr={lr}, γ={gamma}")
    print("=" * 60 + "\n")
    
    for ep in range(num_episodes):
        # 重置环境
        state, _ = env.reset()
        
        states_list, actions_list, rewards_list = [], [], []
        
        done = False
        while not done:
            # 选择动作 (采样)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            log_prob = policy(state_tensor)
            dist = Categorical(torch.exp(log_prob))
            action = dist.sample()
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # 存储数据
            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)
            
            state = next_state
        
        # 计算回报并更新策略
        returns = agent.compute_returns(rewards_list, gamma=gamma)
        
        loss = agent.train_step(
            torch.FloatTensor(np.array(states_list)),
            torch.LongTensor(actions_list),
            returns
        )
        
        episode_reward = sum(rewards_list)
        episode_rewards.append(episode_reward)
        
        # 打印统计信息
        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            running_reward = avg_reward if running_reward is None else \
                0.99 * running_reward + 0.01 * avg_reward
            
            print(f"Episode {ep:4d} | Reward: {episode_reward:6.1f} | "
                  f"Avg(50): {avg_reward:6.1f} | Loss: {loss:.4f}")
        
        # 检查是否达到目标 (CartPole 目标：平均 500 分，连续 100 轮)
        if ep > 100 and np.mean(episode_rewards[-100:]) >= 500:
            print(f"\n✅ 在第 {ep} 轮达到目标！(平均奖励 >= 500)")
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, linewidth=1.2, label='Episode Reward')
    
    # 滑动平均
    window = min(50, len(episode_rewards) // 2)
    if window > 0:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window, len(episode_rewards)+1), smoothed, 
                'r-', linewidth=2, label=f'Smoothed (window={window})')
    
    # 目标线
    plt.axhline(y=500, color='g', linestyle='--', linewidth=2, label='Target (500)')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title(f'REINFORCE Training Curve - {env_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = f'/media/dhu/dhu/work/git/rl_tutorial/chapters/plots/reinforce_curve_{env_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 训练曲线已保存：{save_path}")
    
    plt.close()
    
    # 保存策略网络
    torch.save(policy.state_dict(), 
               f'/media/dhu/dhu/work/git/rl_tutorial/chapters/code/policy_{env_name}.pt')
    print(f"📦 策略已保存：policy_{env_name}.pt")
    
    return policy, episode_rewards


def evaluate_policy(env, policy, num_episodes=100):
    """评估训练好的策略"""
    
    env = gym.make(env) if isinstance(env, str) else env
    
    total_reward = 0
    max_length = 0
    
    print(f"\n🎯 评估 {num_episodes} 轮:")
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_length = 0
        
        done = False
        while not done:
            with torch.no_grad():  # 推理模式，不需要梯度
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                probs = policy(state_tensor)
                action = torch.argmax(probs, dim=-1).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_length += 1
        
        total_reward += episode_length - 1  # CartPole: 不计算初始状态
        max_length = max(max_length, episode_length - 1)
    
    avg_reward = total_reward / num_episodes
    
    print(f"  ✅ 平均寿命：{avg_reward:.2f} 步")
    print(f"  🏆 最长记录：{max_length} 步")
    
    return avg_reward


if __name__ == '__main__':
    import torch.nn.functional as F
    
    # 训练示例 (CartPole)
    policy, rewards = train_reinforce(
        env_name="CartPole-v1",
        num_episodes=2000,
        hidden_dim=128,
        lr=1e-2,
        gamma=0.99
    )
    
    # 评估策略
    evaluate_policy("CartPole-v1", policy)
