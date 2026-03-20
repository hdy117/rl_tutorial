## 第五章：Deep Q-Network (DQN) - 用神经网络替代 Q 表

### 全局导航图

```text
                    +------------------+
                    |  延迟奖励 / 长远目标 |
                    +---------+--------+
                              |
                              v
                     Value / Bellman / TD
                              |
                              v
                    Q-Learning / DQN

                              |
              +---------------+----------------+
              |                                |
              v                                v
     直接优化策略需求                    连续动作 + 样本效率需求
              |                                |
              v                                v
     Policy Gradient                    DDPG -> TD3 -> SAC
              |
              v
        Actor-Critic
              |
              v
            PPO
              |
      +-------+--------+
      |                |
      v                v
  LLM 后训练        经典控制继续
      |
      v
 RLHF / PPO for LLMs
      |
      v
   GRPO
      |
      v
Outcome -> Process -> Verifiable Reward
```

> 你在这里：主干 -> DQN

---

## 问题：Q-Table 的致命限制

Q-Learning 的更新规则本身是正确的：

```math
Q(s,a) \leftarrow Q(s,a) + \alpha \underbrace{\left[ r + \gamma \max_{a'}Q(s',a') - Q(s,a) \right]}_{\text{TD Error}}
```

**公式变量说明**：

| 符号 | 含义 | 说明 |
|------|------|------|
| $Q(s,a)$ | 当前状态-动作对的价值 | 在状态 $s$ 执行动作 $a$ 的预期回报 |
| $\alpha$ | 学习率 | 控制每次更新的步长，如 0.001 |
| $r$ | 即时奖励 | 执行动作 $a$ 后环境返回的奖励 |
| $\gamma$ | 折扣因子 | 0-1 之间，表示对未来的重视程度 |
| $s'$ | 下一状态 | 执行动作 $a$ 后转移到的状态 |
| $\max_{a'}Q(s',a')$ | 下一状态的最大 Q 值 | 在 $s'$ 选择最优动作能获得的值 |

### Q-Table 的三个致命限制

**限制 1：状态空间必须有限且可枚举**

```text
CartPole 状态: s = [cart_position, cart_velocity, pole_angle, pole_velocity]
              每个都是连续值（浮点数）
              → 理论上无限多状态，无法建表
```

**限制 2：离散化导致指数爆炸**

既然状态是连续的，一个想法是把它切成格子变成离散的：

```
cart_position 范围 [-4.8, 4.8]，切成 100 格 → 每格 0.096 单位
cart_velocity 范围 [-∞, +∞]，截断到 [-10, 10]，切成 100 格
pole_angle    范围 [-0.418, 0.418] 弧度，切成 100 格  
pole_velocity 范围截断到 [-10, 10]，切成 100 格
```

**问题：维度的诅咒 (Curse of Dimensionality)**

每个维度单独看只有 100 格，但组合起来：

```
状态数 = 100 × 100 × 100 × 100 = 100^4 = 100,000,000
                                        （一亿个状态！）
```

**存储需求计算**：
- 每个状态存 2 个动作的 Q 值（CartPole 左右）
- 每个 Q 值是 float（4 字节）
- 总内存 = 100,000,000 × 2 × 4 字节 = **800 MB**

这还只是 4 维的 CartPole！如果是 Atari 游戏（84×84 像素 = 7056 维）：
```
状态数 = 100^7056 = 10^14112  （宇宙原子数只有 10^80）
```

**结论：离散化在连续状态空间根本不可行。**

**限制 3：没有泛化能力**

- 学过状态 A，不代表会状态 B
- 即使 A 和 B 几乎一样（如 cart_position 差 0.001），也得分开学

> **核心矛盾：Q-Learning 的更新规则完美，但 Q-Table 无法扩展到真实任务。**

---

## 解决方案：用神经网络替代 Q-Table

### 核心发明：$Q(s,a) \rightarrow Q(s,a; \theta)$

彻底换掉存储方式：

```
旧方式：Q[s][a] = float          (查表，每个状态单独存储)
新方式：Q(s, a; θ) = NN(s)[a]    (函数逼近，参数共享)
```

**关键洞察**：
- 输入状态 $s$ → 神经网络输出所有动作的 Q 值
- 参数 $\theta$ 被训练成"压缩版的 Q-Table"
- 相似状态共享参数 → **自动泛化**

### CartPole 示例

```
输入:  s = [x, x_dot, theta, theta_dot]  (4 维连续向量)
          ↓
      [神经网络]
          ↓
输出:  [Q(s,左), Q(s,右)]  (2 个动作的 Q 值)
```

不需要离散化，直接端到端学习。

---

## DQN 的训练目标

### Loss 函数

```math
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \underbrace{\left( \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)}_{\text{TD Error}} \right)^2}_{\text{MSE Loss}} \right]
```

**公式变量说明**：

| 符号 | 含义 | 说明 |
|------|------|------|
| $L(\theta)$ | 损失函数 | 衡量预测 Q 值与目标 Q 值的差距 |
| $\theta$ | 在线网络参数 | 正在被训练的网络参数 |
| $\theta^-$ | 目标网络参数 | 定期从在线网络复制，训练时固定 |
| $\mathbb{E}_{(s,a,r,s') \sim D}$ | 从重放缓冲区采样求期望 | $D$ 是存储历史经验的回放缓冲区 |
| $Q(s, a; \theta)$ | 在线网络预测 | 当前网络对 $(s,a)$ 的 Q 值估计 |
| $Q(s', a'; \theta^-)$ | 目标网络预测 | 目标网络对下一状态的 Q 值估计 |
| $y = r + \gamma \max_{a'}Q(s',a';\theta^-)$ | Bellman 目标 | 这是"老师"给的目标值 |

### 两个关键技巧

**1. 经验回放 (Replay Buffer)**

```python
# 问题：连续经验高度相关，神经网络学不好
s1 → s2 → s3 → s4  (状态几乎一样)

# 解决：存储经验，随机采样打散
buffer = [(s1,a1,r1,s2), (s5,a5,r5,s6), (s2,a2,r2,s3), ...]
batch = random_sample(buffer, batch_size=32)
```

**为什么有效**：让数据更接近独立同分布 (i.i.d.)，符合神经网络训练假设。

**2. 目标网络 (Target Network)**

```python
# 问题：如果 target 自己也变，训练目标乱飘
# Q(s,a) 要去追 r + γ max Q(s',a') 
# 但 Q(s',a') 也在变！就像追一个移动的靶子

# 解决：固定目标网络，定期更新
θ^- ← θ  (每 N 步复制一次在线网络参数)
# 训练时 θ^- 不变，只有 θ 更新
```

**为什么有效**：稳定训练目标，让学习更像监督学习。

---

## DQN 算法流程

```
初始化: 在线网络 Q(·;θ), 目标网络 Q(·;θ⁻), 回放缓冲区 D

for episode = 1, 2, ..., M:
    获取初始状态 s₁
    
    for t = 1, 2, ..., T:
        # 1. ε-贪婪选择动作
        以 ε 概率随机选动作 aₜ
        以 1-ε 概率选 aₜ = argmaxₐ Q(sₜ, a; θ)
        
        # 2. 执行动作，观察转移
        执行 aₜ，获得奖励 rₜ，观察下一状态 sₜ₊₁
        
        # 3. 存储经验
        D ← D ∪ {(sₜ, aₜ, rₜ, sₜ₊₁, done)}
        
        # 4. 训练（如果缓冲区足够）
        从 D 随机采样 mini-batch {(sⱼ, aⱼ, rⱼ, sⱼ', doneⱼ)}
        
        对每个样本 j:
            如果 doneⱼ 为真:
                yⱼ = rⱼ                              (终止状态，无未来)
            否则:
                yⱼ = rⱼ + γ · maxₐ' Q(sⱼ', a'; θ⁻)   (Bellman 目标)
        
        # 5. 梯度下降更新
        Loss = (1/N) Σⱼ (yⱼ - Q(sⱼ, aⱼ; θ))²
        θ ← θ - α · ∇θ Loss
        
        # 6. 定期更新目标网络
        每隔 C 步: θ⁻ ← θ
        
        sₜ ← sₜ₊₁
```

---

## 为什么 DQN 有效？

### 验证标准 1：保留 Bellman 结构

DQN 的训练目标仍然是 bootstrap 的 Bellman target：

```math
y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
```

**关键**：target 的数学形式和 Q-Learning 完全一样，只是 $Q$ 从查表变成了神经网络输出。

### 验证标准 2：解决可扩展性

| 场景 | Q-Table | DQN |
|------|---------|-----|
| CartPole (4维连续) | ❌ 无法直接用 | ✅ 直接处理 |
| Atari (84×84像素) | ❌ 完全不可能 | ✅ 端到端学习 |
| 泛化到新状态 | ❌ 不会 | ✅ 自动泛化 |

### 验证标准 3：稳定性补丁是必要的

没有 Replay Buffer 和 Target Network 会怎样？

```
问题 1: 样本相关 → 神经网络过拟合局部模式
问题 2: 目标漂移 → 训练震荡，甚至发散
```

这两个技巧不是"锦上添花"，而是让 DQN 能训练的**工程必要性**。

---

## 最小可行代码示例

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

# 1. 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # 输出每个动作的 Q 值
        )
    
    def forward(self, state):
        return self.net(state)  # shape: (batch, action_dim)

# 2. 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# 3. DQN 训练器
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # 在线网络（训练）和目标网络（稳定目标）
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.epsilon = 1.0  # 探索率
        
    def select_action(self, state):
        # ε-贪婪策略
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state)
            return q_values.argmax().item()
    
    def train_step(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return
        
        # 采样
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 当前 Q 值: Q(s, a)
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 目标 Q 值: r + γ * max Q(s', a') （注意用 target_net）
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 损失函数 (MSE)
        loss = nn.MSELoss()(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        """定期复制在线网络参数到目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())
```

---

## DQN 的本质压缩

```
DQN = Q-Learning 的更新规则 + 神经网络的泛化能力

Replay Buffer   = 打散相关性（工程必要性）
Target Network  = 稳定训练目标（工程必要性）
```

### 历史定位

> **DQN 完成了价值函数 RL 从"小表格玩具"到"高维感知任务"的跨越。**

- Q-Learning：解决了"**怎么学动作价值**"
- DQN：解决了"**在大状态空间里怎么表示动作价值**"

---

## DQN 的局限与下一步

### 局限

1. **只能处理离散动作** - CartPole（左右）、Atari（按键）OK，但连续控制（如电机扭矩）不行
2. **样本效率不高** - 需要与环境交互百万步才能收敛
3. **训练仍不稳定** - 超参数敏感，调参困难

### 自然的下一步

> **"既然我最终想要的是策略 π(a|s)，能不能别绕道学 Q，直接优化策略本身？"**

这就逼出了下一章：**Policy Gradient**。

---

## 本章速查表

| 概念 | 一句话解释 |
|------|-----------|
| **$Q(s,a;\theta)$** | 神经网络参数化的动作价值函数 |
| **$\theta$ / $\theta^-$** | 在线网络参数 / 目标网络参数 |
| **Replay Buffer** | 存储历史经验，随机采样打散相关性 |
| **Target Network** | 定期更新的固定目标，稳定训练 |
| **$y = r + \gamma \max Q(s',a';\theta^-)$** | Bellman 目标（教师信号） |
| **Loss = $(y - Q(s,a;\theta))^2$** | MSE 损失，让预测接近目标 |

### 关键公式

```math
\text{预测: } Q(s, a; \theta)
```

```math
\text{目标: } y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
```

```math
\text{损失: } L(\theta) = \mathbb{E}[(y - Q(s, a; \theta))^2]
```

---

## 下一章预告：Policy Gradient

既然 Q-Learning 和 DQN 都是学"动作价值"，那有没有可能**直接优化策略本身**？

这就是 Policy Gradient 要解决的问题。
