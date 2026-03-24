#!/usr/bin/env python3
"""
Actor-Critic 算法完整实现
结合策略梯度 (Actor) 和价值函数 (Critic)，显著降低方差

核心思想：
- Actor: π_θ(a|s) - 策略网络，输出动作分布
- Critic: V_w(s) - 价值网络，评估状态价值
- Advantage: A(s,a) = r + γ*V(s') - V(s) (TD error)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    """Actor-Critic 联合网络"""
    
    def __init__(self, state_dim, action_dim, discrete=True, hidden_dim=128):
        super().__init__()
        
        self.discrete = discrete
        
        # 共享特征提取层 (可选，也可以 Actor/Critic 分开)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if discrete:
            # Actor: 离散动作 -> logits
            self.actor = nn.Linear(hidden_dim, action_dim)
            
            # Critic: 状态价值 (标量输出)
            self.critic = nn.Linear(hidden_dim, 1)
        else:
            # Actor: 连续动作 -> μ + log_std
            self.actor_mu = nn.Linear(hidden_dim, action_dim)
            self.actor_logstd = nn.Linear(hidden_dim, action_dim)
            
            # Critic
            self.critic = nn.Linear(hidden_dim, 1)
        
        # 正交初始化，帮助训练稳定
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """同时输出 Actor 和 Critic
        
        Returns:
            - 离散：(logits, value)
            - 连续：(mu, std, value)
        """
        features = self.shared(state)
        
        if self.discrete:
            logits = self.actor(features)
            value = self.critic(features).squeeze(-1)
            return logits, value
        
        else:
            mu = self.actor_mu(features)
            log_std = self.actor_logstd(features)
            
            # 限制 log_std 范围，防止数值问题
            log_std = torch.clamp(log_std, -20, 2)
            
            value = self.critic(features).squeeze(-1)
            return mu, torch.exp(log_std), value
    
    def get_action(self, state, deterministic=False):
        """采样动作 (用于与环境交互)
        
        Args:
            state: [batch_size, state_dim] 或 [state_dim]
            deterministic: 是否使用 argmax/mean (推理模式)
        
        Returns:
            action, log_prob
        """
        # 确保 batch 维度
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if self.discrete:
            logits, _ = self.forward(state)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
            
            log_prob = dist.log_prob(action).squeeze(-1)
            return action.detach(), log_prob.detach()
        
        else:
            mu, std, _ = self.forward(state)
            
            if deterministic:
                action = torch.tanh(mu)  # 映射到 (-1, 1)
            else:
                dist = Normal(mu, std)
                action = dist.sample()
                action = torch.tanh(action)  # 映射到合法范围
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.detach(), log_prob.detach()


class ActorCriticAgent:
    """Actor-Critic 训练器"""
    
    def __init__(self, state_dim, action_dim, discrete=True, 
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99):
        self.model = ActorCritic(state_dim, action_dim, discrete)
        
        # 分开优化 Actor 和 Critic，可以有不同的学习率
        actor_params = [p for name, p in self.model.named_parameters() 
                       if 'actor' in name]
        critic_params = [p for name, p in self.model.named_parameters() 
                        if 'critic' in name or ('mu' not in name and 'logstd' not in name)]
        
        # 注意：对于离散情况，shared 层参数分配需要小心处理
        
        self.actor_optimizer = optim.Adam(actor_params + critic_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
        
        self.gamma = gamma
    
    def compute_td_error(self, states, actions, rewards, next_states, dones):
        """计算 TD error (Advantage 估计)"""
        
        # 获取当前价值估计
        if self.model.discrete:
            _, current_value = self.model(states)
            _, next_value = self.model(next_states)
        else:
            mu, std, _ = self.model(states)
            _, _, next_value = self.model(next_states)
        
        # TD target: r + γ * V(s')
        td_target = rewards + self.gamma * (1 - dones.float()) * next_value
        
        # TD error: A ≈ r + γ*V(s') - V(s)
        td_error = td_target - current_value
        
        return td_error.detach(), td_target


def train_actor_critic(env_name="CartPole-v1",
                      num_episodes=500,
                      hidden_dim=128,
                      lr_actor=3e-4,
                       lr_critic=1e-3,
                      gamma=0.99,
                      discrete=True):
    """
    训练 Actor-Critic
    
    Args:
        env_name: Gymnasium 环境名称
        num_episodes: 训练轮数
        hidden_dim: 隐藏层维度
        lr_actor/critic: Actor/Critic 学习率
        gamma: 折扣因子
        discrete: 是否离散动作空间
    """
    
    # 初始化环境和智能体
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = ActorCriticAgent(state_dim, action_dim, discrete, 
                            lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma)
    
    # 训练统计
    episode_rewards = []
    
    print(f"\n🚀 开始训练 Actor-Critic ({env_name})")
    print("=" * 60)
    print(f"参数：hidden_dim={hidden_dim}, γ={gamma}")
    print(f"学习率: Actor={lr_actor}, Critic={lr_critic}")
    print("=" * 60 + "\n")
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        
        states_batch, actions_batch, rewards_batch = [], [], []
        
        done = False
        while not done:
            # 采样动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = agent.model.get_action(state_tensor)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states_batch.append(state)
            actions_batch.append(action)
            rewards_batch.append(reward)
            
            state = next_state
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states_batch))
        actions = torch.LongTensor(actions_batch) if discrete else torch.FloatTensor(actions_batch)
        rewards = torch.FloatTensor(rewards_batch).unsqueeze(-1)
        
        dones = torch.FloatTensor([1.0])  # episode 结束
        
        # ========== Critic Loss (TD error minimization) ==========
        td_error, td_target = agent.compute_td_error(states, actions, rewards, states, dones)
        
        critic_loss = nn.functional.mse_loss(td_target, 
                                             td_target.detach() + td_error)
        
        # ========== Actor Loss (Policy Gradient with Advantage) ==========
        advantage = td_error  # TD error as advantage estimate
        
        if discrete:
            logits, _ = agent.model(states)
            dist = Categorical(torch.softmax(logits, dim=-1))
            log_probs = dist.log_prob(actions)
        else:
            mu, std, _ = agent.model(states)
            dist = Normal(mu, std)
            actions_clamped = torch.clamp(actions, -1, 1)
            log_probs = dist.log_prob(actions_clamped).sum(dim=-1)
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # ========== 更新参数 ==========
        agent.actor_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        agent.actor_optimizer.step()
        
        episode_reward = sum(rewards_batch)
        episode_rewards.append(episode_reward)
        
        # 打印统计信息
        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            
            print(f"Episode {ep:4d} | Reward: {episode_reward:6.1f} | "
                  f"Avg(50): {avg_reward:6.1f} | "
                  f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")
        
        # 检查目标达成
        if ep > 100 and np.mean(episode_rewards[-100:]) >= 500:
            print(f"\n✅ 在第 {ep} 轮达到目标！")
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, linewidth=1.2, label='Episode Reward')
    
    window = min(50, len(episode_rewards) // 2)
    if window > 0:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window, len(episode_rewards)+1), smoothed, 
                'r-', linewidth=2, label=f'Smoothed (w={window})')
    
    plt.axhline(y=500, color='g', linestyle='--', linewidth=2, label='Target (500)')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    title = f"Actor-Critic Training ({env_name}, {'Discrete' if discrete else 'Continuous'})"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = f'/media/dhu/dhu/work/git/rl_tutorial/chapters/plots/ac_curve_{env_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 训练曲线已保存：{save_path}")
    
    # 保存模型
    torch.save(agent.model.state_dict(),
               f'/media/dhu/dhu/work/git/rl_tutorial/chapters/code/ac_{env_name}.pt')
    print(f"📦 模型已保存：ac_{env_name}.pt")
    
    return agent, episode_rewards


def evaluate_actor_critic(env_name="CartPole-v1", model=None, num_episodes=100):
    """评估训练好的 Actor-Critic 策略"""
    
    env = gym.make(env_name) if isinstance(env_name, str) else env_name
    
    # 加载模型 (如果未提供)
    if model is None:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # 简化：假设离散，实际需要根据环境类型判断
        model = ActorCritic(state_dim, action_dim, discrete=True)
        
        checkpoint_path = f'/media/dhu/dhu/work/git/rl_tutorial/chapters/code/ac_{env_name}.pt'
        try:
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"✅ 加载模型：{checkpoint_path}")
        except FileNotFoundError:
            print("⚠️ 未找到预训练模型，使用随机策略")
    
    total_reward = 0
    
    print(f"\n🎯 评估 Actor-Critic ({num_episodes} 轮):")
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _ = model.get_action(state_tensor, deterministic=True)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            state = next_state
        
        total_reward += reward
    
    avg_reward = total_reward / num_episodes
    print(f"  ✅ 平均奖励：{avg_reward:.2f}")
    
    return avg_reward


if __name__ == '__main__':
    # 训练示例 (CartPole - Discrete)
    agent, rewards = train_actor_critic(
        env_name="CartPole-v1",
        num_episodes=500,
        hidden_dim=128,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        discrete=True
    )
    
    # 评估策略
    evaluate_actor_critic("CartPole-v1", agent.model)
