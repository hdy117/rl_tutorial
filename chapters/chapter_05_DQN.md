## 第五章：Deep Q-Network (DQN) - 用神经网络替代 Q 表

### 📍 全局导航图

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

> **你在这里：主干 → DQN**（用神经网络实现 Q-Learning）

---

## 一、Problem：Q-Table 为什么死在真实世界？

### 核心矛盾

Q-Learning 的更新规则完美无缺：

$$\color{firebrick} Q(s,a) \leftarrow Q(s,a) + \alpha \underbrace{\left[ r + \gamma \max_{a'}Q(s',a') - Q(s,a) \right]}_{\text{TD Error}}$$

**但存储方式彻底失效**。

### 维度诅咒 (Curse of Dimensionality)

```
┌─────────────────────────────────────────────────────┐
│  CartPole 状态: s = [x, x_dot, theta, theta_dot]   │
│             每个都是连续值（浮点数）                  │
│              → 理论上无限多状态，无法建表           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  尝试离散化：                                        │
│  x       ∈ [-4.8, +4.8]     → 切 100 格             │
│  x_dot   ∈ [-∞, +∞]        → 截断，切 100 格         │
│  theta   ∈ [-0.42, +0.42]   → 切 100 格              │
│  θ_dot   ∈ [-∞, +∞]        → 截断，切 100 格         │
└─────────────────────────────────────────────────────┘

状态总数 = 100 × 100 × 100 × 100 = 10⁸（一亿个状态！）

内存需求：10⁸ × 2 actions × 4 bytes = 800 MB
```

**Atari 游戏更绝望**：
- 输入：84×84×3 RGB 像素 = **7,056 维连续空间**
- 如果切 100 格 → $100^{7056} \approx 10^{14112}$ 个状态
- **宇宙原子数只有 $10^{80}$**

### 三个致命限制总结

| 问题 | Q-Table | DQN 解决方式 |
|------|---------|-------------|
| **连续状态空间** | ❌ 无法直接表示 | ✅ 神经网络输入任意维度 |
| **状态泛化** | ❌ 学 A ≠ 会 B（即使 A≈B） | ✅ 参数共享自动泛化 |
| **可扩展性** | ❌ 指数爆炸 | ✅ 线性扩展（参数量固定） |

---

## 二、Starting Point：我们有什么工具？

### Q-Learning 的精华部分

$$\color{blue} y = r + \gamma \max_{a'}Q(s',a') \quad \text{(Bellman Target)}$$

**这个目标值本身没错**。问题只是 $Q$ 从"查表"变成了"函数逼近"。

### 神经网络的天然适配性

```
┌─────────────────────────────────────────────────────┐
│  神经网络本质：参数化函数 Q(s, a; θ)                │
│                                                     │
│  输入 s → 隐藏层变换 → 输出 Q(s,·) 对所有动作       │
│                                                     │
│  θ 通过梯度下降训练，让 Q(s,a;θ) ≈ 真实 Q-Table   │
└─────────────────────────────────────────────────────┘

视觉化：Q-Table vs Q-Network

┌──────────────────┐     ┌──────────────────────┐
│    Q-Table       │     │    Q-Network (θ)     │
├──────────────────┤     ├──────────────────────┤
│  [s1][a1] → val1 │     │   s = [x₁, x₂, ...] │
│  [s1][a2] → val2 │     │         ↓            │
│  [s2][a1] → val3 │     │    ┌─────────┐      │
│  [s2][a2] → val4 │     │    │ Neural  │      │
│       ...        │     │    │ Network │      │
│                  │     │    └─────────┘      │
│  查表 O(1)        │     │         ↓            │
│  无泛化           │     │  Q(s,a₁), Q(s,a₂)...│
│  空间爆炸         │     │                     │
│                  │     │  参数共享 → 自动泛化 │
└──────────────────┘     └──────────────────────┘
```

---

## 三、Invention：DQN 如何从第一原理推导？

### Step 1: Axioms（不可约的事实）

**公理 1**: Q-Learning 的 Bellman Target 是 RL 收敛的核心
$$y = r + \gamma \max_{a'}Q(s',a')$$

**公理 2**: 神经网络可以拟合任意连续函数（Universal Approximation Theorem）
$$f(x; \theta) \approx g(x), \quad \forall g \text{ (足够复杂的网络)}$$

**公理 3**: Q-Table 本质是离散函数 $Q: S \times A \to \mathbb{R}$

### Step 2: Contradictions（矛盾）

```
┌───────────────────────────────────────────────────┐
│  如果 Q-Learning 正确，神经网络可以拟合任意函数    │
│  那为什么不能直接用神经网络做 Q-Table？          │
│                                                   │
│  → 试了发现训练不稳定、不收敛                    │
│  → 需要额外技巧（Replay Buffer + Target Network）│
└───────────────────────────────────────────────────┘
```

### Step 3: Solution Path（唯一合理的解决路径）

#### 【核心原理图】DQN 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    DQN Training Loop                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌─────────────────────┐  │
│   │ Environment│ → │ Agent    │ → │ Replay Buffer       │  │
│   │ (Gym)     │ ← │ (Policy) │ ← │ (随机采样打散相关性) │  │
│   └──────────┘    └──────────┘    └─────────────────────┘  │
│        |               |                  │                │
│        v               v                  v                │
│   Environment      ε-贪婪策略           Batch (s,a,r,s')   │
│   Step()          select_action()       ← random sample    │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Training Step                         │  │
│   ├─────────────────────────────────────────────────────┤  │
│   │                                                     │  │
│   │  Current Net (θ):   Q(s, a; θ)                     │  │
│   │                    ↓                                │  │
│   │  Target Net (θ⁻):   y = r + γ·maxₐ'Q(s',a';θ⁻)     │  │
│   │                    ↓                                │  │
│   │  Loss: L(θ) = E[(y - Q(s,a;θ))²]                   │  │
│   │                    ↓                                │  │
│   │  Backprop → θ ← θ - α·∇θL                         │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │         Target Network Update (每 C 步)            │  │
│   │              θ⁻ ← θ (参数复制)                     │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 【原理图】Replay Buffer 如何打散相关性

```text
传统 RL（时序训练）：
┌─────────────────────────────────────────┐
│ s₁ → a₁ → r₁ → s₂ → a₂ → r₂ → s₃ ...   │
│ ↑         ↑         ↑                   │
│ └─────────┴─────────┘                    │
│     状态高度相关！                       │
│     神经网络过拟合局部模式               │
└─────────────────────────────────────────┘

DQN（Replay Buffer）：
┌─────────────────────────────────────────┐
│ 存储: [(s₁,a₁,r₁,s₂), ..., (sₜ,aₜ,rₜ,sₜ₊₁)] │
│      ↓                                  │
│ 采样: random_sample(buffer, batch_size) │
│      → [s₅, s₁₀, s₂, s₈, ...]           │
│      打散时序相关性 ✅                   │
└─────────────────────────────────────────┘

数学本质：让数据更接近 i.i.d.（独立同分布）假设
```

#### 【原理图】Target Network 为什么必要？

```text
问题场景（无 Target Net）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t=0: Q(s,a) = 1.0, target y = r + γ·Q(s',a') = 5.0
     → Loss = (5.0 - 1.0)² = 16
     → θ 更新，Q 值变成 2.0
     
t=1: Q(s,a) = 2.0, target y = r + γ·Q(s',a') = 7.0 
     （Q(s',a') 自己也变了！target 漂移）
     → Loss = (7.0 - 2.0)² = 25
     → θ 更新，但目标也在动 → 震荡/发散

有 Target Net：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t=0: Q(s,a;θ) = 1.0, target y = r + γ·Q(s',a';θ⁻) = 5.0
     （θ⁻固定，target 稳定）
     → θ 更新
     
t=1: Q(s,a;θ) = 2.0, target y = r + γ·Q(s',a';θ⁻) = 5.0
     （θ⁻仍然固定！target 不变）
     → θ 继续稳定学习
     
每 C 步：θ⁻ ← θ（缓慢移动 target，避免漂移）
```

### Step 4: Compression Mechanisms（如何可扩展）

**参数共享压缩**：
- Q-Table：每个状态单独存储 $O(|S| \cdot |A|)$
- DQN：固定参数量 $\theta$，与状态空间大小无关

**泛化能力**：
- 相似状态 → 相似的神经网络输出
- 见过状态 A，自动会类似的状态 B

### Step 5: Verification（独立重推检验）

遮住答案，尝试自己推导：

1. **Q-Learning 的 Bellman Target 是什么？** 
   - $y = r + \gamma \max_{a'}Q(s',a')$
2. **神经网络如何表示 Q-Table？**
   - $Q(s,a;\theta)$ 输入状态输出动作价值
3. **训练目标如何定义？**
   - $\min_\theta \mathbb{E}[(y - Q(s,a;\theta))^2]$
4. **为什么需要 Replay Buffer？**
   - 打散时序相关性，符合 i.i.d.假设
5. **为什么需要 Target Network？**
   - 固定训练目标，避免 bootstrap 漂移

---

## 四、Verification：DQN 算法验证

### Loss 函数详解

$$\color{firebrick} L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

**公式拆解**：

| 符号 | 含义 | 为什么这样设计 |
|------|------|----------------|
| $L(\theta)$ | MSE Loss | 监督学习标准形式 |
| $\mathbb{E}_{(s,a,r,s') \sim D}$ | 从重放缓冲区采样 | i.i.d.假设，打散相关性 |
| $Q(s, a; \theta)$ | **在线网络**预测当前值 | 正在被训练的网络 |
| $Q(s', a'; \theta^-)$ | **目标网络**预测未来值 | 定期更新，保持稳定 |
| $\gamma$ | 折扣因子 (0-1) | 控制未来奖励的权重 |

### 算法流程（伪代码）

```python
# 初始化
online_net = QNetwork(θ)          # 在线网络（训练）
target_net = QNetwork(θ⁻)         # 目标网络（固定）
buffer = ReplayBuffer(capacity=1M)

for episode in range(M):
    s = env.reset()
    
    for t in range(T):
        # 1. ε-贪婪选择动作
        a = random if ε else argmaxₐ Q(s, a; θ)
        
        # 2. 执行动作，观察转移
        s', r, done = env.step(a)
        
        # 3. 存储经验
        buffer.add((s, a, r, s', done))
        
        # 4. 训练（如果缓冲区足够）
        if len(buffer) > BATCH_SIZE:
            batch = random_sample(buffer, BATCH_SIZE)
            
            # 计算 target y
            for (sj, aj, rj, sj', done_j) in batch:
                if done_j:
                    yj = rj
                else:
                    yj = rj + γ * maxₐ' Q(sj', a'; θ⁻)  # ← Target Net!
            
            # 梯度下降
            Loss = MSE(Q(batch.s, batch.a; θ), batch.y)
            θ ← θ - α * ∇θLoss
        
        # 5. 定期更新目标网络（每 C 步）
        if step % C == 0:
            target_net.load_state_dict(online_net.state_dict())
            
        s = s'
```

---

## 五、Example：最小可行代码实现

### 神经网络架构

```python
import torch
import torch.nn as nn
from collections import deque
import random

class QNetwork(nn.Module):
    """Q-Network: 输入状态 → 输出所有动作的 Q 值"""
    
    def __init__(self, state_dim=4, hidden_dims=[128, 128], action_dim=2):
        super().__init__()
        
        # 构建多层感知机
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(state_dim, h_dim))
            layers.append(nn.ReLU())
            state_dim = h_dim
        
        layers.append(nn.Linear(state_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        """state: (batch_size, state_dim) → Q-values: (batch_size, action_dim)"""
        return self.net(state)


class ReplayBuffer:
    """经验回放缓冲区：存储历史经验，随机采样打散相关性"""
    
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)  # 环形缓冲区
    
    def add(self, state, action, reward, next_state, done):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样 batch（打散时序相关性）"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),    # (B, state_dim)
            torch.LongTensor(actions),   # (B,)
            torch.FloatTensor(rewards),  # (B,)
            torch.FloatTensor(next_states), # (B, state_dim)
            torch.FloatTensor(dones)     # (B,)
        )


class DQNAgent:
    """DQN Agent：整合网络、优化器、训练逻辑"""
    
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.action_dim = action_dim
        
        # 在线网络（训练）和目标网络（稳定目标）
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())  # 初始同步
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=lr, weight_decay=1e-4
        )
        
        # 经验回放
        self.buffer = ReplayBuffer()
        
        # ε-贪婪参数（探索率，随时间衰减）
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995  # 每步乘以这个数
    
    def select_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 探索
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)  # (1, state_dim)
            q_values = self.q_net(state)                   # (1, action_dim)
            return q_values.argmax().item()                 # exploitation
    
    def train_step(self, batch_size=32):
        """训练一步：从 buffer 采样 → 计算 Loss → 梯度下降"""
        if len(self.buffer) < batch_size:
            return None
        
        # 1. 采样 mini-batch
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 2. 当前 Q 值：Q(s, a; θ)
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 3. 目标 Q 值：y = r + γ * maxₐ' Q(s', a'; θ⁻)
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1)[0]  # max over actions
            target_q = rewards + self.gamma * next_q_max * (1 - dones)
        
        # 4. MSE Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # 5. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)  # 梯度裁剪
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """复制在线网络参数到目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon * self.epsilon_decay
        )
```

### 训练循环示例

```python
# 环境设置（以 CartPole 为例）
import gymnasium as gym

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

for episode in range(500):
    state = env.reset()[0]
    total_reward = 0
    
    for t in range(1000):  # CartPole max steps
        # Select and execute action
        action = agent.select_action(state)
        next_state, reward, terminated, _, _ = env.step(action)
        
        # Store transition
        agent.buffer.add(state, action, reward, next_state, terminated)
        
        # Train step
        loss = agent.train_step(batch_size=32)
        
        state = next_state
        total_reward += reward
        
        if terminated:
            break
    
    # Decay exploration rate
    agent.decay_epsilon()
    
    # Update target network every 10 steps
    if episode % 10 == 0:
        agent.update_target_network()
    
    print(f"Episode {episode}: Reward = {total_reward:.1f}, Epsilon = {agent.epsilon:.3f}")

env.close()
```

---

## DQN 的本质压缩

### 三行总结

$$\color{firebrick} \text{DQN} = \underbrace{\text{Q-Learning Bellman Target}}_{\text{核心算法}} + \underbrace{\text{神经网络的泛化能力}}_{\text{可扩展性}} + \underbrace{\text{Replay Buffer + Target Net}}_{\text{工程稳定技巧}}$$

### 历史定位

> **DQN 完成了价值函数 RL 从"小表格玩具"到"高维感知任务"的跨越。**

- **Q-Learning**：解决了"**怎么学动作价值**"（Bellman Equation）
- **DQN**：解决了"**在大状态空间里怎么表示动作价值**"（神经网络函数逼近）

### DQN 解决了什么，留下了什么

| ✅ 已解决 | ❌ 仍待解决 |
|----------|------------|
| 连续状态空间 | 只能处理离散动作 |
| 自动泛化到新状态 | 样本效率低（百万步） |
| 端到端学习（像素→动作） | 训练不稳定，超参数敏感 |

---

## 第六章预告：Policy Gradient

> **"既然我最终想要的是策略 π(a|s)，能不能别绕道学 Q，直接优化策略本身？"**

这就是 Policy Gradient 要解决的问题。

