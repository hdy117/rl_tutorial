#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) 简化实现
核心创新：Clip Loss + 约束更新步长

为什么需要 PPO？
- Actor-Critic 方差低，但单次更新可能破坏策略
- PPO 通过裁剪机制限制更新幅度，更稳定

关键公式:
L^CLIP(θ) = E[min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)]

其中 r_t(θ) = π_θ(a|s) / π_{θ_old}(a|s)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class PPOLoss:
    """PPO 代理损失函数 (Clip Loss)"""
    
    def __init__(self, eps_clip=0.2):
        self.eps_clip = eps_clip
    
    def compute(self, old_log_probs, new_log_probs, advantages):
        """
        计算 PPO Clip Loss
        
        Args:
            old_log_probs: 旧策略的对数概率 [batch_size]
            new_log_probs: 新策略的对数概率 [batch_size]
            advantages: Advantage 估计 [batch_size]
        
        Returns:
            loss: 标量损失值 (负号用于最小化)
        """
        # 计算比率 r_t(θ) = exp(log π_new - log π_old)
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        
        # 无裁剪的损失 (标准 PG)
        surr1 = ratio * advantages
        
        # 裁剪后的比率，限制在 [1-ε, 1+ε]
        ratio_clipped = torch.clamp(ratio, 
                                   1-self.eps_clip, 1+self.eps_clip)
        
        # 裁剪后的损失
        surr2 = ratio_clipped * advantages
        
        # 取最小值：保守更新，防止策略破坏
        loss = -torch.min(surr1, surr2).mean()
        
        return loss


class PPO:
    """PPO 简化实现 (单网络版本)"""
    
    def __init__(self, model, state_dim, action_dim, discrete=True,
                 lr=3e-4, eps_clip=0.2, gamma=0.99):
        self.model = model
        self.discrete = discrete
        self.gamma = gamma
        
        # 存储旧策略数据 (用于计算 advantage)
        self.old_data = []
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = PPOLoss(eps_clip)
    
    def store_transition(self, states, actions, log_probs, values):
        """存储一次 episode 的数据"""
        self.old_data.append({
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'values': values
        })
    
    def compute_returns(self, rewards):
        """计算折扣回报 G_t (Monte Carlo)"""
        returns = []
        G = 0
        
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        return torch.tensor(returns, dtype=torch.float32)
    
    def train_step(self, advantages, old_values):
        """训练一步 (所有 stored data 一起更新)"""
        
        # 合并所有 episode 的数据
        all_states = torch.cat([d['states'] for d in self.old_data], dim=0)
        all_actions = torch.cat([d['actions'] for d in self.old_data], dim=0)
        old_log_probs = torch.cat([d['log_probs'] for d in self.old_data], dim=0)
        
        # 重新计算新策略的对数概率和值估计
        if self.discrete:
            logits, values = self.model(all_states)
            new_dist = Categorical(torch.softmax(logits, dim=-1))
            new_log_probs = new_dist.log_prob(all_actions)
        else:
            mu, std, values = self.model(all_states)
            new_dist = Normal(mu, std)
            
            # 注意：如果 action 经过 tanh，需要雅可比校正 (简化版忽略)
            actions_clamped = torch.clamp(all_actions, -1, 1)
            new_log_probs = new_dist.log_prob(actions_clamped).sum(dim=-1)
        
        # PPO Loss
        ppo_loss = self.loss_fn.compute(
            old_log_probs, new_log_probs, advantages
        )
        
        # Value Function Loss (MSE)
        value_loss = nn.functional.mse_loss(values.squeeze(), 
                                           old_values)
        
        # Entropy Bonus (鼓励探索，防止过早收敛)
        entropy = new_dist.entropy().mean() if self.discrete else \
                  new_dist.entropy().sum(dim=-1).mean()
        
        # Total Loss = Policy Loss + 0.5*Value Loss - β*Entropy
        total_loss = ppo_loss + 0.5 * value_loss - 0.01 * entropy
        
        # 更新参数
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return ppo_loss.item(), value_loss.item(), -entropy.item()


def train_ppo(env_name="CartPole-v1",
             num_episodes=200,
             hidden_dim=128,
             lr=3e-4,
             eps_clip=0.2,
             gamma=0.99):
    """
    PPO 训练主循环
    
    Args:
        env_name: Gymnasium 环境名称
        num_episodes: 训练轮数 (PPO 样本效率高，通常 100-300 轮)
        hidden_dim: 隐藏层维度
        lr: 学习率
        eps_clip: Clip 参数 (通常 0.1-0.2)
        gamma: 折扣因子
    """
    
    # 初始化环境和模型
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim)  # logits (离散情况)
    )
    
    agent = PPO(model, state_dim, action_dim, discrete=True,
               lr=lr, eps_clip=eps_clip, gamma=gamma)
    
    # 训练统计
    episode_rewards = []
    
    print(f"\n🚀 开始训练 PPO ({env_name})")
    print("=" * 60)
    print(f"参数：hidden_dim={hidden_dim}, ε={eps_clip}, γ={gamma}")
    print("PPO 特点: Clip Loss + 约束更新步长 ✅")
    print("=" * 60 + "\n")
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        
        states_list, actions_list, log_probs_list, rewards_list = [], [], [], []
        
        done = False
        while not done:
            # 选择动作 (采样)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                logits = model(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states_list.append(state_tensor)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            
            state = next_state
        
        # 计算 Advantage (简化：用 G_t - mean(G_t))
        returns = agent.compute_returns(rewards_list)
        
        # Critic: 估计状态价值 (实际应该训练一个独立的价值网络)
        # 这里简化为：用 returns 的滑动平均作为 baseline
        if len(episode_rewards) > 0:
            value_estimate = np.mean(returns.numpy())
        else:
            value_estimate = 0
        
        advantages = returns - value_estimate
        
        # 存储旧策略数据
        agent.store_transition(
            torch.cat(states_list, dim=0),
            torch.stack(actions_list),
            torch.cat(log_probs_list),
            returns
        )
        
        # PPO 训练一步 (使用 stored data)
        ppo_loss, value_loss, entropy = agent.train_step(advantages, returns)
        
        episode_reward = sum(rewards_list)
        episode_rewards.append(episode_reward)
        
        # 清空存储数据
        agent.old_data = []
        
        # 打印统计信息
        if ep % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            
            print(f"Episode {ep:4d} | Reward: {episode_reward:6.1f} | "
                  f"Avg(20): {avg_reward:6.1f} | "
                  f"PPO Loss: {ppo_loss:.4f} | Entropy: {-entropy:.3f}")
        
        # 检查目标达成 (PPO 通常更快收敛)
        if ep > 50 and np.mean(episode_rewards[-50:]) >= 500:
            print(f"\n✅ 在第 {ep} 轮达到目标！(PPO 样本效率高)")
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, linewidth=1.2, label='Episode Reward')
    
    window = min(20, len(episode_rewards) // 3)
    if window > 0:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window, len(episode_rewards)+1), smoothed, 
                'r-', linewidth=2, label=f'Smoothed (w={window})')
    
    plt.axhline(y=500, color='g', linestyle='--', linewidth=2, label='Target (500)')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    title = f"PPO Training Curve - {env_name} (ε={eps_clip})"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = f'/media/dhu/dhu/work/git/rl_tutorial/chapters/plots/ppo_curve_{env_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 训练曲线已保存：{save_path}")
    
    # 保存模型
    torch.save(model.state_dict(),
               f'/media/dhu/dhu/work/git/rl_tutorial/chapters/code/ppo_{env_name}.pt')
    print(f"📦 模型已保存：ppo_{env_name}.pt")
    
    return model, episode_rewards


def compare_methods(env_name="CartPole-v1"):
    """比较 REINFORCE vs Actor-Critic vs PPO"""
    
    print("\n" + "=" * 70)
    print("📊 POLICY GRADIENT 方法对比 (CartPole)")
    print("=" * 70)
    
    methods = [
        ("REINFORCE", {"num_episodes": 2000, "lr": 1e-2}),
        ("Actor-Critic", {"num_episodes": 500, "lr_actor": 3e-4, 
                         "lr_critic": 1e-3}),
        ("PPO", {"num_episodes": 200, "lr": 3e-4})
    ]
    
    results = []
    
    for method_name, train_kwargs in methods:
        print(f"\n🔥 {method_name}:")
        
        try:
            if method_name == "REINFORCE":
                from reinforce import train_reinforce
                _, rewards = train_reinforce(env_name, **train_kwargs)
            
            elif method_name == "Actor-Critic":
                from actor_critic import train_actor_critic
                _, rewards = train_actor_critic(env_name, **train_kwargs, discrete=True)
            
            elif method_name == "PPO":
                _, rewards = train_ppo(env_name, **train_kwargs)
            
            # 计算统计指标
            min_episodes = min(200, len(rewards))
            final_avg = np.mean(rewards[-min_episodes:])
            
            # 收敛速度 (达到平均 500 分所需的轮数)
            converged_at = None
            for i in range(min_episodes, len(rewards)):
                if np.mean(rewards[i-min_episodes:i]) >= 500:
                    converged_at = i
                    break
            
            results.append({
                'method': method_name,
                'final_avg': final_avg,
                'converged_at': converged_at,
                'rewards': rewards
            })
            
            print(f"  ✅ 最终平均 (最后{min_episodes}轮): {final_avg:.1f}")
            if converged_at:
                print(f"  🏆 收敛点：第 {converged_at} 轮")
            else:
                print(f"  ⚠️ 未达到目标 (500 分)")
        
        except Exception as e:
            print(f"  ❌ 训练失败: {e}")
    
    # 可视化对比图
    plt.figure(figsize=(12, 6))
    
    colors = ['red', 'blue', 'green']
    for i, (res, color) in enumerate(zip(results, colors)):
        rewards_smoothed = np.convolve(res['rewards'], 
                                       np.ones(50)/50, mode='valid')
        
        if res['method'] == "REINFORCE":
            plt.plot(np.arange(len(rewards_smoothed)), rewards_smoothed[:50],
                    color=color, linewidth=2, label=res['method'])
        else:
            plt.plot(np.arange(1, len(res['rewards'])+1), res['rewards'],
                    color=color, alpha=0.6, linewidth=1.5, 
                    label=f"{res['method']} (raw)")
            
            smoothed = np.convolve(res['rewards'], np.ones(20)/20, mode='valid')
            plt.plot(np.arange(len(smoothed)), smoothed,
                    color=color, linewidth=2, alpha=0.8)
    
    plt.axhline(y=500, color='k', linestyle='--', linewidth=1.5, label='Target (500)')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title("Policy Gradient 方法对比：REINFORCE vs Actor-Critic vs PPO",
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = '/media/dhu/dhu/work/git/rl_tutorial/chapters/plots/pg_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图已保存：{save_path}")
    
    # 总结表
    print("\n" + "=" * 70)
    print("📋 方法对比总结:")
    print("=" * 70)
    print(f"{'方法':<15} {'收敛轮数':<12} {'最终平均':<12} {'样本效率'}")
    print("-" * 70)
    
    for res in results:
        converged_str = f"{res['converged_at']}" if res['converged_at'] else "未收敛"
        
        efficiency = "🐢慢" if res['method'] == "REINFORCE" else \
                    ("⚡快" if res['method'] == "PPO" else "🚶中")
        
        print(f"{res['method']:<15} {converged_str:<12} {res['final_avg']:.1f}<{'':<8} {efficiency}")
    
    print("=" * 70)


if __name__ == '__main__':
    # 训练示例 (CartPole)
    model, rewards = train_ppo(
        env_name="CartPole-v1",
        num_episodes=200,
        hidden_dim=128,
        lr=3e-4,
        eps_clip=0.2,
        gamma=0.99
    )
