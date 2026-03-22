# 第五章：DQN - 用神经网络破解维度诅咒 🔥

## 🎯 第一性学习 spine: `problem -> starting point -> invention -> verification -> example`

---

## 一、Problem：为什么 Q-Table 死在真实世界？

### 🔴 核心矛盾：连续 vs 离散

**Q-Learning 的 Bellman Target 完美无缺：**

```math
y = r + \gamma \max_{a'}Q(s',a')
```

**但存储方式彻底崩溃！**

---

### 📊 问题 1: 维度诅咒（Curse of Dimensionality）

#### CartPole：看似简单，实则致命

```text
状态空间：s = [x, x_dot, θ, θ_dot]
          ↓
每个都是连续浮点数 → 理论上无限多状态！

尝试离散化（切 bin）：
    x      ∈ [-4.8, +4.8]     → 100 格
    x_dot  ∈ [-∞, +∞]         → 截断，100 格
    θ      ∈ [-0.42, +0.42]   → 100 格
    θ_dot  ∈ [-∞, +∞]         → 截断，100 格

状态总数 = 100⁴ = 100,000,000 (一亿！)
Q-Table 大小 = 10⁸ × 2 actions × 4 bytes ≈ 800 MB
```

#### Atari：彻底绝望

```text
像素输入：84×84×3 RGB = 21,168 维连续空间

理论状态数：
    - 每个像素 ∈ [0, 255]
    - 256²¹⁶⁸ ≈ 10⁵⁰⁰⁰⁰
    
即使每维只切 100 格：
    状态数 = 100²¹⁶⁸

对比：
    ✅ 宇宙原子数      ≈ 10⁸⁰
    ❌ Atari Q-Table   ≈ 10⁵⁰⁰⁰⁰
```

**结论：** Q-Table 在真实世界根本不存在！

---

### 📊 问题 2: 没有泛化能力（State Generalization）

#### Q-Table 的"死板"行为

```text
状态 A: [x=2.34, x_dot=-0.89, θ=0.12, θ_dot=0.03]
         │
         ▼
    Q-table[ A ][ left ] = 5.2

状态 B: [x=2.36, x_dot=-0.87, θ=0.11, θ_dot=0.04]
         │ (和 A 几乎一样！)
         ▼
    Q-table[ B ][ left ] = ??? (未见过 → 初始值 0)

结果：
- Agent 在 A 学会"向左好"
- 遇到相似的 B 却表现得像新手
```

**为什么？** Q-Table 是查表，A ≠ B（即使相似）→ **无推理能力！**

---

### 📊 问题 3: 可扩展性爆炸

#### 空间复杂度对比

| 算法 | 空间复杂度 | CartPole (100 格) | Atari (像素) |
|------|-----------|------------------|-------------|
| **Q-Table** | O(\|S\|×\|A\|) | 800 MB ✅ | 10⁵⁰⁰⁰⁰ ❌ |
| **DQN** | O(参数量 θ) | ~200 KB ✅ | ~4 MB ✅ |

#### 为什么神经网络是线性的？

```text
Q-Table:
    输入维度 ↑ → bin 数指数增长
    10 维 × 100 = 10¹⁰ entries
    20 维 × 100 = 10²⁰ entries ← 爆炸！

DQN:
    输入维度 ↑ → 第一层权重线性增加
    4 维 → [4→128] = 512 参数
    21,168 维 → conv layer ≈ 1M 参数 ← 可控！
```

**核心洞察：** Q-Table 是**记忆型**（记住所有状态），DQN 是**推理型**（学习规律）！

---

### 📈 问题对比图：Q-Table vs DQN

```text
┌─────────────────────────────┬─────────────────────────────┐
│      Q-Table (记忆)          │        DQN (推理)            │
├─────────────────────────────┼─────────────────────────────┤
│ 连续空间                     │ 连续空间                     │
│ ❌ 必须离散化                │ ✅ 直接接受连续输入          │
│     ↓                        │                              │
│     精度损失/空间爆炸        │     平滑函数                 │
├─────────────────────────────┼─────────────────────────────┤
│ 泛化能力                     │ 泛化能力                     │
│ ❌ A ≠ B (查表)              │ ✅ A ≈ B → f(A) ≈ f(B)      │
│     ↓                        │                              │
│     无推理                   │     参数共享 + 连续性        │
├─────────────────────────────┼─────────────────────────────┤
│ 可扩展性                     │ 可扩展性                     │
│ ❌ O(|S|×|A|) 指数增长       │ ✅ O(θ) 线性扩展             │
│     ↓                        │                              │
│     维度诅咒                 │     函数逼近压缩             │
└─────────────────────────────┴─────────────────────────────┘

一句话：Q-Learning = "记住所有答案"，DQN = "学会推理规律" 🔥
```

---

## 二、Starting Point：我们有什么工具？

### 🧠 Q-Learning 的精华（完美保留）

```math
y = r + \gamma \max_{a'}Q(s',a')
```

**这个 Bellman Target 本身没错！** 问题只是 $Q$ 从"查表"变成"函数逼近"。

---

### 🧠 神经网络：参数化函数逼近器

#### 数学定义

```text
Q-Network: f(s; θ) : ℝⁿ → ℝᵐ

输入：s ∈ ℝⁿ (状态向量，n 维连续空间)
      │
      ▼
   多层非线性变换
      │
      ▼
输出：Q(s,·) ∈ ℝᵐ (m 个动作的 Q 值)

关键参数：
    n = 状态空间维度 (CartPole:4, Atari:21,168)
    m = 动作数量 (CartPole:2, Atari:18)
    θ = 网络所有权重和偏置 (固定参数量！)
```

---

### 📊 Q-Table vs Q-Network：视觉对比

#### 结构对比图

```text
┌───────────────────────┐         ┌───────────────────────┐
│     Q-Table           │         │    Q-Network (θ)      │
├───────────────────────┤         ├───────────────────────┤
│                       │         │                       │
│  s = "状态索引"       │         │  s = [x, x_dot...]    │
│        │              │         │        │              │
│        ▼              │         │        ▼              │
│  ┌─────────┐          │         │   ┌──────────┐        │
│  │查表 O(1)│          │         │   │ 前向传播│        │
│  └────┬────┘          │         │   └────┬─────┘        │
│       │ Q(s,a)        │         │        │ Q(s,·;θ)     │
│       ▼               │         │        ▼              │
│  无泛化               │         │  自动泛化 ✅          │
│  空间爆炸 ❌          │         │  参数共享 ✅          │
└───────────────────────┘         └───────────────────────┘
```

#### 完整训练循环对比图

```text
Q-Table（简单但脆弱）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Env
      │ s (离散索引)
      ▼
  ┌───────┐
  │ Q-Tab │ ← 查表 O(1)，无泛化
  └───┬───┘
      │ Q(s,·)
      ▼
   Policy → a → Env (循环)

空间复杂度：O(|S|×|A|) ❌


Q-Network（强大但需稳定）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Env
      │ s (连续向量)
      ▼
  ┌─────────┐
  │Network  │ ← 前向传播，自动泛化 ✅
  │ θ       │
  └────┬────┘
       │ Q(s,·;θ)
       ▼
    Policy → a → Env (循环)

空间复杂度：O(θ) ✅
但需要 Replay Buffer + Target Net 稳定训练！
```

---

## 三、Invention：DQN 如何从第一原理推导？

### 🎯 DQN = Q-Learning + 神经网络 + 工程技巧

#### 整体架构全景图

```text
┌─────────────────────────────────────────────────────────────┐
│                   DQN Training Loop (完整循环)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐      s       ┌──────────┐                  │
│   │  Env     │───────────►  │ Policy   │                   │
│   │(Gym)     │              │ε-Greedy  │                   │
│   └────┬─────┘              └────┬─────┘                   │
│        │ a                       │ a                        │
│        ▼                         ▼                          │
│   ┌──────────────────────────────────────┐                │
│   │         s', r, done                  │                │
│   │          │                           │                │
│   │          ▼                           │                │
│   │    ┌───────────┐                     │                │
│   │    │Replay     │◄──── (s,a,r,s',done) │               │
│   │    │Buffer     │   容量 ~1M           │               │
│   │    │(环形队列) │                    │                │
│   │    └─────┬─────┘                     │                │
│   │          │ random sample             │                │
│   │          ▼                           │                │
│   │  ┌──────────────────────────┐       │                │
│   │  │ Training Step (每 C 步)   │◄──────┘                │
│   │  ├───────────────────────┬───┤                        │
│   │  │ Target Net (θ⁻)固定   │ Loss=(y-Q)²                 │
│   │  │ Current Net (θ)训练   │ Backprop                   │
│   │  └───────────────────────┴───┘                        │
│   └─────────────────────────────────────────────────────────┘
│                                                             │
│         Target: y = r + γ·maxₐ'Q(s',a';θ⁻)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

关键设计：两个工程技巧解决理论完美但实践崩溃的问题！
```

---

### 🔍 Step 1: Axioms（不可约的事实）

#### 三个公理

**公理 1**: Bellman Target 是 RL 收敛的核心

```math
y = r + \gamma \max_{a'} Q(s', a')
```

**公理 2**: 神经网络可以拟合任意连续函数

```math
f(x; \theta) \approx g(x), \quad \forall g \text{ (足够复杂)}
```

**公理 3**: Q-Table 本质是离散函数

```math
Q: S × A → ℝ
```

---

### 🧠 公理 3 的深层含义（关键！）

#### Q-Table 不是"表格"，而是**函数映射**！

```text
数学定义：
    Q: S × A → ℝ
    (s, a) ↦ Q(s,a)

两个表示法：
    1. Q-Table: Q(s,a) = table[s][a]      ← 离散存储
    2. DQN:     Q(s,a; θ) = f(s,a)        ← 连续函数

本质相同！都是学一个函数 Q(s,a) ≈ Q^*(s,a)
区别：用表格还是神经网络表示这个函数！
```

#### 第一性原理验证（遮住答案问自己）

```text
问题链：
    1. Q-Learning 要学的是什么？ 
       → 一个函数 Q: S×A → ℝ
    
    2. 这个函数的输入输出是什么？
       → (s,a) 实数/离散 → Q 值（实数）
    
    3. 表格和神经网络都能表示这样的函数吗？
       → ✅ 都能！
    
    4. 那为什么选神经网络？
       → 因为泛化能力 + 可扩展性
    
结论：DQN = "用神经网络实现 Q-Learning"是必然的！
```

---

### 🚨 Step 2: Contradictions（矛盾）

#### 逻辑推导链

```text
Premise A: Bellman Target 正确
    y = r + γ·maxₐ'Q(s',a')

Premise B: 神经网络可以拟合任意函数
    f(x;θ) ≈ g(x)

Premise C: Q-Table 本质是函数
    Q: S×A → ℝ

─────────────────────────────
理论结论：神经网络应该完美替代 Q-Table！
实践测试（1990s）：直接组合 → ❌ 训练崩溃！
```

#### 为什么崩溃？三个致命问题

**问题 1**: Target 漂移（Moving Target）🎯

```text
y = r + γ·maxₐ'Q(s',a';θ)

Q-net 更新 → θ 变化 → y 也变化！
就像追一个不断移动的靶子 🏹
```

**问题 2**: 样本相关性（Correlated Samples）📈

```text
传统训练：s₁→s₂→s₃→s₄... (时序强相关)

神经网络假设数据 i.i.d.
→ 过拟合局部模式 → 无法泛化
```

**问题 3**: 自举偏差（Bootstrapping Bias）🔄

```text
Q(s,a) ← r + γ·Q(s',a')

左边和右边都用同一个网络！
→ 误差累积，正反馈放大 → 震荡发散
```

---

### 📊 矛盾可视化：Target 漂移问题

```text
无 Target Net（崩溃）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=0: Q(s,a)=1.0, target y = r+γ·Q(s',a') = 5.0
     Loss = (5-1)² = 16 → θ 更新 → Q=2.0
     
t=1: Q(s,a)=2.0, target y = r+γ·Q(s',a') = 7.0 ← y 变了！
     Loss = (7-2)² = 25 → θ 更新 → Q=3.0
     ...震荡发散 ❌


有 Target Net（稳定）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=0: Q(s,a;θ)=1.0, target y = r+γ·Q(s',a';θ⁻) = 5.0
     (θ⁻固定！y 稳定) → θ 更新
     
t=1: Q(s,a;θ)=2.0, target y = r+γ·Q(s',a';θ⁻) = 5.0 ← y 不变！
     (θ⁻仍然固定) → θ 继续稳定学习
     
每 C 步：θ⁻ ← θ (缓慢移动靶子) ✅
```

---

### 🔧 Step 3: Solution Path（唯一合理的解决路径）

#### DQN 的两个工程技巧

**技巧 1**: Replay Buffer — 打散相关性

```text
传统 RL:
    s₁→a₁→r₁→s₂→a₂→r₂→s₃ (时序强相关 ❌)

DQN:
    存储：[(s₁,a₁,r₁,s₂), ..., (sₜ,aₜ,rₜ,sₜ₊₁)]
         ↓
    采样：random_sample(buffer, batch_size)
         → [s₅, s₁₀, s₂, s₈, ...] (i.i.d. ✅)

数学本质：让数据接近 i.i.d.假设！
```

**技巧 2**: Target Network — 固定训练目标

```text
双网络设计：
    Current Net (θ):      实时预测，持续训练
    Target Net (θ⁻):      固定参数，提供稳定 target
    
更新策略：
    θ⁻ ← θ (每 C 步复制，缓慢移动)
    
为什么有效？
    - y = r + γ·maxₐ'Q(s',a';θ⁻) → θ⁻固定 → target 稳定
    - Current Net 学习稳定的目标 → 收敛！
```

---

### 📊 Replay Buffer vs Target Network 作用对比图

```text
┌─────────────────────────────┬─────────────────────────────┐
│   Replay Buffer             │   Target Network           │
├─────────────────────────────┼─────────────────────────────┤
│                             │                             │
│ 问题：样本相关性            │ 问题：Target 漂移          │
│ ❌ s₁→s₂→s₃强相关           │ ❌ θ 变化 → y 变化          │
│                             │                             │
│ 解决：随机采样              │ 解决：双网络分离            │
│ ✅ random_sample()          │ ✅ Current vs Target        │
│                             │                             │
│ 效果：i.i.d.假设接近        │ 效果：target 稳定           │
│     符合神经网络训练要求    │     Current Net 可收敛      │
│                             │                             │
└─────────────────────────────┴─────────────────────────────┘

两者缺一不可！Replay Buffer 解决数据质量，Target Net 解决目标稳定性！
```

---

### 📊 Step 4: Compression Mechanisms（如何可扩展）

#### 参数共享压缩机制

```text
Q-Table:
    每个状态单独存储 → O(|S|×|A|)
    CartPole (10⁸ states): ~800 MB
    
DQN:
    参数共享：同一个 θ 处理所有状态 → O(θ)
    CartPole (~200K params): ~200 KB
    Atari (~1M params): ~4 MB

压缩率：从 800MB → 200KB = **4000x 压缩！**
```

#### 泛化能力机制

```text
相似状态 A ≈ B:
    Q-Table: f(A) ≠ f(B) (查表，无推理) ❌
    DQN:     f(A) ≈ f(B) (连续函数 ✅)

为什么？神经网络是插值器！
见过 A → 自动会 B, C, D... (参数共享)
```

---

### 🔍 Step 5: Verification（独立重推检验）

#### 遮住答案，自己推导

**Q1**: Q-Learning 的 Bellman Target 是什么？  
→ $y = r + \gamma \max_{a'}Q(s',a')$

**Q2**: 神经网络如何表示 Q-Table？  
→ $Q(s,a;\theta)$：输入状态，输出动作价值

**Q3**: 训练目标如何定义？  
→ $\min_\theta \mathbb{E}[(y - Q(s,a;\theta))^2]$ (MSE Loss)

**Q4**: 为什么需要 Replay Buffer？  
→ 打散时序相关性 → 符合 i.i.d.假设

**Q5**: 为什么需要 Target Network？  
→ 固定训练目标 → 避免 bootstrap 漂移

---

## 四、Verification：DQN 算法验证

### 📐 Loss 函数详解

```math
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
```

#### 公式拆解表

| 符号 | 含义 | 为什么设计？ |
|------|------|-------------|
| $L(\theta)$ | MSE Loss | 监督学习标准形式 ✅ |
| $\mathbb{E}_{\sim D}$ | Replay Buffer 采样 | i.i.d.假设，打散相关性 ✅ |
| $Q(s,a;\theta)$ | Current Net (θ) | 正在训练的目标网络 ⚡ |
| $Q(s',a';\theta^-)$ | Target Net (θ⁻) | 固定参数，稳定 target 🎯 |
| $\gamma$ | 折扣因子 (0-1) | 控制未来奖励权重 ⏳ |

---

### 📋 算法伪代码（完整流程）

```python
# 初始化
current_net = QNetwork(θ)          # Current Net，持续训练
target_net = QNetwork(θ⁻)          # Target Net，固定参数
buffer = ReplayBuffer(capacity=1M) # 环形缓冲区

for episode in range(M):
    s = env.reset()
    
    for t in range(T):
        # 1. ε-Greedy 选择动作
        a = random_action() if ε else argmaxₐ Q(s, a; θ)
        
        # 2. 执行动作，观察转移
        s', r, done = env.step(a)
        
        # 3. 存储经验到 buffer
        buffer.add((s, a, r, s', done))
        
        # 4. 训练（buffer 足够时）
        if len(buffer) > BATCH_SIZE:
            batch = random_sample(buffer, BATCH_SIZE)
            
            # 计算 target y (用 Target Net!)
            for (sj, aj, rj, sj', done_j) in batch:
                yj = rj + γ * maxₐ' Q(sj', a'; θ⁻) if not done_j else rj
            
            # MSE Loss + Backprop
            loss = MSE(Q(batch.s, batch.a; θ), batch.y)
            θ ← θ - α ∇θLoss
        
        # 5. 每 C 步更新 Target Net
        if step % C == 0:
            target_net.load_state_dict(current_net.state_dict())
        
        s = s'
```

---

## 五、Example：最小可行代码实现

### 🧠 Q-Network 架构（PyTorch）

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Q-Network: 输入状态 → 输出所有动作的 Q 值"""
    
    def __init__(self, state_dim=4, hidden_dims=[128, 128], action_dim=2):
        super().__init__()
        
        # 构建 MLP：输入→隐藏层×N→输出
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(state_dim, h_dim))
            layers.append(nn.ReLU())      # 激活函数
            state_dim = h_dim
        
        layers.append(nn.Linear(state_dim, action_dim))  # 输出层
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        """state: (B, n) → Q-values: (B, m)"""
        return self.net(state)
```

---

### 📦 ReplayBuffer 实现

```python
from collections import deque
import random

class ReplayBuffer:
    """经验回放缓冲区：存储历史，随机采样打散相关性"""
    
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)  # 环形队列
    
    def add(self, state, action, reward, next_state, done):
        """存储一条经验 (s,a,r,s',done)"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样 batch（打散时序相关性！）"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # 解包成 tensor
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),      # (B, n)
            torch.LongTensor(actions),     # (B,)
            torch.FloatTensor(rewards),    # (B,)
            torch.FloatTensor(next_states),# (B, n)
            torch.FloatTensor(dones)       # (B,)
        )
```

---

### 🤖 DQNAgent 完整实现

```python
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    """DQN Agent：整合网络、优化器、训练逻辑"""
    
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # 双网络设计
        self.q_net = QNetwork(state_dim, action_dim)      # Current Net (θ)
        self.target_net = QNetwork(state_dim, action_dim) # Target Net (θ⁻)
        
        # 初始同步
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # 优化器：Adam 适合 RL
        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=lr, weight_decay=1e-4
        )
        
        # Replay Buffer
        self.buffer = ReplayBuffer()
        
        # ε-Greedy 参数（探索率衰减）
        self.epsilon = 1.0      # 初始全探索
        self.epsilon_min = 0.05 # 最小探索率
        self.epsilon_decay = 0.995 # 每步衰减
    
    def select_action(self, state):
        """ε-Greedy: ε%随机探索，(1-ε)%最大化 Q"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Exploration
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state)                    # (1, m)
            return q_values.argmax().item()                 # Exploitation
    
    def train_step(self, batch_size=32):
        """训练一步：采样→计算 target→Loss→Backprop"""
        if len(self.buffer) < batch_size:
            return None
        
        # 1. 随机采样 mini-batch (打散相关性！)
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 2. 当前 Q 值：Q(s, a; θ)
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 3. Target Q 值：y = r + γ * maxₐ' Q(s', a'; θ⁻)
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1)[0]  # max over actions
            target_q = rewards + self.gamma * next_q_max * (1 - dones)
        
        # 4. MSE Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # 5. Backprop + 梯度裁剪（防止爆炸）
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """每 C 步：复制 Current Net → Target Net"""
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率：从 100% → ε_min"""
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon * self.epsilon_decay
        )
```

---

### 🎮 训练循环示例（CartPole）

```python
import gymnasium as gym

# 环境设置
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

for episode in range(500):
    state = env.reset()[0]
    total_reward = 0
    
    for t in range(1000):
        # Select & execute
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
    
    # Decay exploration
    agent.decay_epsilon()
    
    # Update target every 10 episodes
    if episode % 10 == 0:
        agent.update_target_network()
    
    print(f"Episode {episode}: Reward={total_reward:.1f}, Epsilon={agent.epsilon:.3f}")

env.close()
```

---

## DQN 的本质压缩（三行总结）

### 🎯 DQN = 核心算法 + 函数逼近 + 工程稳定技巧

```math
\text{DQN} = \underbrace{\text{Bellman Target}}_{\text{Q-Learning 精髓}} + 
            \underbrace{\text{神经网络泛化}}_{\text{破解维度诅咒}} + 
            \underbrace{\text{Replay Buffer + Target Net}}_{\text{稳定训练}}
```

---

### 📚 历史定位

> **DQN 完成了价值函数 RL 从"小表格玩具"到"高维感知任务"的跨越！**

| 算法 | 解决的问题 | 遗留问题 |
|------|-----------|---------|
| **Q-Learning** | ✅ 怎么学动作价值 (Bellman Equation) | ❌ 只能处理离散小状态空间 |
| **DQN** | ✅ 大状态空间里怎么表示 Q 值 (神经网络逼近)<br>✅ 连续输入、自动泛化<br>✅ 像素→动作端到端学习 | ❌ 只能处理离散动作<br>❌ 样本效率低（百万步）<br>❌ 训练不稳定，超参数敏感 |

---

## 🚀 下一章预告：Policy Gradient

> **"既然我最终想要的是策略 π(a\|s)，能不能别绕道学 Q，直接优化策略本身？"**

这就是 Policy Gradient 要解决的问题！🔥

---

## 📊 学习检查清单（Self-Test）

完成本章后，你应该能：

- [ ] **解释为什么 Q-Table 在真实世界失效**（维度诅咒、无泛化、可扩展性差）
- [ ] **推导 DQN 的三个公理**（Bellman Target、Universal Approximation、Q 是函数）
- [ ] **说明两个矛盾**（Target 漂移、样本相关性、自举偏差）
- [ ] **画出 DQN 完整训练循环图**（Env→Policy→Buffer→Training）
- [ ] **解释 Replay Buffer 和 Target Net 的作用**（打散相关性 vs 稳定目标）
- [ ] **独立写出最小可行 DQN 代码**（QNetwork + ReplayBuffer + Agent）
- [ ] **说出 DQN 解决了什么、留下了什么**（连续状态✅，离散动作❌）

---

## 🔥 关键记忆点

```text
核心矛盾：
    Q-Learning 完美 ✅ 但 Q-Table 死掉 ❌ → 需要神经网络替代！

两个工程技巧：
    Replay Buffer = 打散相关性 (i.i.d.假设)
    Target Net = 固定 target (避免移动靶子)

DQN 本质：
    记忆型 → 推理型
    查表问题 → 函数拟合问题
```

