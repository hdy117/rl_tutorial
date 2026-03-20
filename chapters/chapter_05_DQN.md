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
│  CartPole 状态: s = [x, x_dot, theta, theta_dot]    │
│             每个都是连续值（浮点数）                   │
│              → 理论上无限多状态，无法建表             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  尝试离散化：                                        │
│  x       ∈ [-4.8, +4.8]     → 切 100 格             │
│  x_dot   ∈ [-∞, +∞]         → 截断，切 100 格        │
│  theta   ∈ [-0.42, +0.42]   → 切 100 格             │
│  θ_dot   ∈ [-∞, +∞]         → 截断，切 100 格        │
└─────────────────────────────────────────────────────┘

状态总数 = 100 × 100 × 100 × 100 = 10⁸（一亿个状态！）

内存需求：10⁸ × 2 actions × 4 bytes = 800 MB
```

**Atari 游戏更绝望**：
- 输入：84×84×3 RGB 像素 = **7,056 维连续空间**
- 如果切 100 格 → $100^{7056} \approx 10^{14112}$ 个状态
- **宇宙原子数只有 $10^{80}$**

### 三个致命限制总结

#### 🔥 问题 1: 连续状态空间 (Continuous State Space)

**为什么 Q-Table 彻底失效？**

Q-Learning 的 Q-table 是离散映射：每个 `(s, a)` 对对应一个值。但真实世界的状态是连续的：

```text
CartPole 的真实状态:
┌─────────────────────────────────────────────────────┐
│ x       = +2.347812... (小车位置)                    │
│ x_dot   = -0.892345... (速度)                        │
│ theta   = +0.123456... (杆角度)                      │
│ θ_dot   = +0.034567... (角速度)                      │
└─────────────────────────────────────────────────────┘

每个值都是浮点数，理论上：
- x 可以是 [-4.8, +4.8] 范围内的任意实数
- 有无穷多个可能的状态值
- Q-table 需要为每个精确的 (s,a) 对建表 → 不可能！
```

**尝试离散化的后果：**

```text
如果切成 100 格：
┌─────────────────────────────────────────────────────┐
│ x ∈ [-4.8, +4.8]      → 切 100 个 bin               │
│ x_dot ∈ [-∞, +∞]      → 截断，切 100 个 bin         │
│ θ ∈ [-0.42, +0.42]    → 切 100 个 bin               │
│ θ_dot ∈ [-∞, +∞]      → 截断，切 100 个 bin         │
└─────────────────────────────────────────────────────┘

状态总数 = 100⁴ = 100,000,000 (一亿！)
Q-table 大小 = 10⁸ × 2 actions × 4 bytes = 800 MB
```

**问题在哪里？**

| 维度 | Q-Table 的问题 | DQN 的解决 |
|------|---------------|-----------|
| **表示** | 需要离散化 → 信息丢失 | ✅ 神经网络直接接受连续输入 |
| **精度损失** | `x=2.347` 和 `x=2.358` 被划到同一格 → Q 值相同 | ✅ 每个精确输入得到不同输出 |
| **边界效应** | bin 边界的微小变化导致 Q 值突变 | ✅ 连续函数，平滑过渡 |

> **核心矛盾：** 真实世界是连续的，Q-Table 是离散的。离散化要么精度不够（bin 太大），要么空间爆炸（bin 太小）。

---

#### 🔥 问题 2: 状态泛化 (State Generalization)

**为什么 Q-Table 没有泛化能力？**

```text
场景：小车在不同位置但相似状态

┌────────────────────────────────────────────────────────────┐
│ 状态 A: [x=2.34, x_dot=-0.89, θ=0.12, θ_dot=0.03]          │
│         → Q-table 记录 Q(A, left)=5.2                      │
│                                                            │
│ 状态 B: [x=2.36, x_dot=-0.87, θ=0.11, θ_dot=0.04]          │
│         (和 A 几乎一样，只是数值微小差异)                    │
│         → Q-table 记录 Q(B, left)=???                      │
└────────────────────────────────────────────────────────────┘

Q-Table 的行为：
- A 和 B 是不同的键（key）
- 学了 A ≠ 自动知道 B
- 需要分别访问 A 和 B 才能学到两者的价值

结果：
- 训练时可能只见过状态 A，没见过 B
- 遇到 B 时 Q 值还是初始值（通常是 0）
- Agent 在 B 表现得像新手，尽管它已经"学会"了类似情况！
```

**DQN 如何解决？**

```text
神经网络视角：Q(s, a; θ) = f(s) → Q 值

┌─────────────────────────────────────────────────────────┐
│ 状态 A: [2.34, -0.89, 0.12, 0.03]  → 输入网络 → Q≈5.2   │
│                                                         │
│ 状态 B: [2.36, -0.87, 0.11, 0.04]  → 输入网络 → Q≈5.3   │
└─────────────────────────────────────────────────────────┘

关键：神经网络是连续函数！
- A ≈ B（输入相似）→ f(A) ≈ f(B)（输出相似）
- **参数共享**：同一个 θ 处理所有状态
- 学了 A → 自动会类似的 B、C、D...

数学本质：神经网络的插值能力
- Q-Table: 离散点，无法推断中间值
- DQN: 连续曲面，自然泛化到未见区域
```

> **核心矛盾：** Q-Table 是查表（无推理），神经网络是函数逼近（有推理）。泛化能力来自参数共享和连续性。

---

#### 🔥 问题 3: 可扩展性 (Scalability)

**为什么状态爆炸无法承受？**

```text
Q-Table 的空间复杂度：O(|S| × |A|)

场景对比：
┌─────────────────────────────────────────────────────┐
│ CartPole (离散化):                                  │
│ - 10⁸ states × 2 actions = 2×10⁸ entries            │
│ - 内存需求：~800 MB                                 │
│ ✓ 勉强可行                                          │
└─────────────────────────────────────────────────────┘

Atari 游戏 (像素输入):
┌─────────────────────────────────────────────────────┐
│ 输入：84×84×3 RGB = 21,168 维                       │
│ - 每个像素值 ∈ [0, 255]                              │
│ - 理论状态数：256²¹⁶⁸ ≈ 10⁵⁰⁰⁰⁰                     │
│                                                      │
│ 即使只切 100 格每维：                                │
│ - 状态数 = 100²¹⁶⁸                                   │
│ - 宇宙原子数只有 10⁸⁰！                               │
│                                                      │
│ ❌ 彻底不可行                                       │
└─────────────────────────────────────────────────────┘

DQN 的空间复杂度：O(参数量)
┌─────────────────────────────────────────────────────┐
│ Atari DQN (典型架构):                               │
│ - Conv layers: ~1M parameters                       │
│ - Fully connected: ~50K parameters                  │
│ - 总内存：~4 MB                                     │
│ ✓ 与输入维度无关（固定参数量）                        │
└─────────────────────────────────────────────────────┘

对比：
┌─────────────────────────────────────────────────────┐
│ Q-Table:    O(|S|×|A|) → 随状态空间指数增长          │
│ DQN:        O(θ)       → 参数量固定，线性扩展        │
└─────────────────────────────────────────────────────┘
```

**为什么神经网络是线性的？**

```text
Q-Table:
输入维度 ↑ → 需要更多格子 → Q-table 指数增长
10 维 × 100 格 = 10¹⁰ entries
20 维 × 100 格 = 10²⁰ entries ← 爆炸！

DQN:
输入维度 ↑ → 网络第一层权重增加 → 参数量线性增长
4 维输入 → [4→128] = 512 参数
84×84×3=21168 维 → conv layer ≈ 1M 参数 ← 可控！

关键：神经网络通过层级结构压缩信息，而不是存储每个状态
```

> **核心矛盾：** Q-Table 是**记忆型**（记住所有状态），DQN 是**推理型**（学习规律）。可扩展性来自函数逼近而非表格存储。

---

### 三个问题的本质对比

| 问题 | Q-Table（记忆） | DQN（推理） |
|------|----------------|-------------|
| **连续空间** | ❌ 无法表示，必须离散化 → 精度损失或爆炸 | ✅ 神经网络天然接受任意维度连续输入 |
| **泛化能力** | ❌ 状态 A ≠ 状态 B（即使相似），无推理能力 | ✅ 参数共享 + 连续性 → 自动泛化到未见状态 |
| **可扩展性** | ❌ O(\|S\|×\|A\|) 指数爆炸，维度诅咒 | ✅ O(θ) 线性扩展，参数量与状态空间无关 |

> **一句话总结：** Q-Learning 是"**记住所有答案**"，DQN 是"**学会推理规律**"。当世界太大、太连续时，记忆失效，必须推理。

---

## 二、Starting Point：我们有什么工具？

### Q-Learning 的精华部分

$$\color{blue} y = r + \gamma \max_{a'}Q(s',a') \quad \text{(Bellman Target)}$$

**这个目标值本身没错**。问题只是 $Q$ 从"查表"变成了"函数逼近"。

### 神经网络的天然适配性

```
┌─────────────────────────────────────────────────────┐
│  神经网络本质：参数化函数 Q(s, a; θ)                 │
│                                                     │
│  输入 s → 隐藏层变换 → 输出 Q(s,·) 对所有动作        │
│                                                     │
│  θ 通过梯度下降训练，让 Q(s,a;θ) ≈ 真实 Q-Table      │
└─────────────────────────────────────────────────────┘

视觉化：Q-Table vs Q-Network

┌─────────────────────┐     ┌──────────────────────────┐
│      Q-Table        │     │      Q-Network (θ)       │
├─────────────────────┤     ├──────────────────────────┤
│  [s1][a1] → val1    │     │  s = [x₁, x₂, ...]       │
│  [s1][a2] → val2    │     │          ↓               │
│  [s2][a1] → val3    │     │     ┌───────────┐        │
│  [s2][a2] → val4    │     │     │   Neural  │        │
│       ...           │     │     │  Network  │        │
│                     │     │     └───────────┘        │
│  查表 O(1)           │     │          ↓               │
│  无泛化              │     │  Q(s,a₁), Q(s,a₂)...     │
│  空间爆炸            │     │                          │
│                     │     │  参数共享 → 自动泛化      │
└─────────────────────┘     └──────────────────────────┘
```

---

## 三、Invention：DQN 如何从第一原理推导？

### Step 1: Axioms（不可约的事实）

**公理 1**: Q-Learning 的 Bellman Target 是 RL 收敛的核心
$$y = r + \gamma \max_{a'}Q(s',a')$$

**公理 2**: 神经网络可以拟合任意连续函数（Universal Approximation Theorem）
$$f(x; \theta) \approx g(x), \quad \forall g \text{ (足够复杂的网络)}$$

**公理 3**: Q-Table 本质是离散函数 $Q: S \times A \to \mathbb{R}$

**这个公理的深层含义：**

```text
Q-Table 不是"表格"，而是数学上的函数映射！

形式化定义：
┌─────────────────────────────────────────────────────┐
│ Q: S × A → ℝ                                        │
│                                                      │
│ (s, a) ↦ Q(s,a)                                     │
│                                                      │
│ 其中：                                               │
│ - S = {s₁, s₂, ..., sₙ} 状态空间（离散）             │
│ - A = {a₁, a₂, ..., aₘ} 动作空间（离散）             │
│ - ℝ    实数域（Q 值可以是任意实数）                   │
└─────────────────────────────────────────────────────┘

关键洞察：
1. Q-Table 是**函数表示法 1** — 用表格存储所有 (s,a) → Q 的映射
2. 神经网络是**函数表示法 2** — 用参数 θ 编码同样的映射关系
3. **两者本质相同，只是实现方式不同！**

数学等价性：
┌─────────────────────────────────────────────────────┐
│ 表示法 1 (Q-Table):                                  │
│   Q(s,a) = table[s][a]                              │
│                                                      │
│ 表示法 2 (DQN):                                     │
│   Q(s,a; θ) = neural_net(s, a)                      │
│                                                      │
│ 目标相同：Q(s,a) ≈ Q^*(s,a)                         │
└─────────────────────────────────────────────────────┘

为什么是"公理"？（不可约）
- Q-Learning 的本质就是学出一个函数 Q: S×A → ℝ
- **用什么表示这个函数不重要**（表格/网络/公式都行）
- Bellman Optimality 只关心函数的性质，不关心实现形式
- 所以 DQN = "用神经网络实现 Q-Learning"是必然的！

第一性原理验证：
遮住答案问自己：
1. Q-Learning 要学的是什么？ → 一个函数 Q(s,a)
2. 这个函数的输入输出是什么？ → (s,a) → ℝ
3. 表格和神经网络都能表示这样的函数吗？ → ✅ 都能！
4. 那为什么选神经网络？ → 因为泛化能力和可扩展性

结论：DQN 不是新算法，而是 Q-Learning 的"函数逼近版本"。
```

### Step 2: Contradictions（矛盾）

```text
┌───────────────────────────────────────────────────────┐
│ 逻辑推导链：                                           │
│                                                        │
│ Premise A: Q-Learning 的 Bellman Target 正确           │
│     y = r + γ·maxₐ'Q(s',a')                           │
│                                                        │
│ Premise B: 神经网络可以拟合任意连续函数                 │
│     f(x;θ) ≈ g(x)                                     │
│                                                        │
│ Premise C: Q-Table 本质是函数 Q: S×A → ℝ               │
│                                                        │
│ ─────────────────────────────────────────────         │
│ 结论：神经网络应该能完美替代 Q-Table！                   │
│     Q(s,a;θ) ← y                                      │
└───────────────────────────────────────────────────────┘

现实测试（1990s）：直接拿神经网络跑 Q-Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
尝试 1: 最简单的实现
```python
Q_net = NeuralNetwork(input_dim=state, output_dim=action)

for episode in range(1000):
    for t in range(100):
        s, a, r, s', done = env.step()
        
        # Bellman Target（完全照抄 Q-Learning）
        y = r + γ * maxₐ' Q_net(s', a')  # ← 问题在这里！
        
        # 梯度下降训练
        loss = (y - Q_net(s, a))²
        Q_net.backward(loss)
```

结果：❌ **不收敛，震荡发散**

为什么？
┌─────────────────────────────────────────────────────┐
│ 问题 1: Target 漂移（Moving Target）                 │
│   y = r + γ·maxₐ'Q(s',a';θ)                         │
│   Q_net 更新 → θ 变化 → y 也变化！                   │
│   就像追一个不断移动的靶子 🎯                        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 问题 2: 样本相关性（Correlated Samples）             │
│   s₁→s₂→s₃→s₄... (时序强相关)                       │
│   神经网络假设数据 i.i.d.                            │
│   → 过拟合局部模式，无法泛化                         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 问题 3: 自举偏差（Bootstrapping Bias）               │
│   Q(s,a) ← r + γ·Q(s',a')                           │
│   左边和右边都用同一个网络！                          │
│   → 误差累积，正反馈放大                             │
└─────────────────────────────────────────────────────┘

矛盾的本质：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
理论层面：DQN = Q-Learning + 神经网络（完美）
实践层面：直接组合 → 训练崩溃（灾难）

→ **需要额外的工程技巧来稳定训练！**
```

### Step 3: Solution Path（唯一合理的解决路径）

#### 【核心原理图】DQN 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────
│                           DQN Training Loop                                  
│                                                                              
│  ┌──────────────────┐         ┌──────────────────┐        ┌────────────────┐ 
│  │   Environment    │◄────────│  Policy (ε-Greedy)       │  Replay Buffer  │ 
│  │      (Gym)       │          │                   │        │              │ 
│  │                  │─────────►│         Q(s,·;θ)    │◄─────│  (s,a,r,s',d) │ 
│  │   观察状态 s     │动作 a    │                   │        │   容量 ~1M    │ 
│  └────────┬─────────┘          └────────┬──────────┘        └───────┬──────┘ 
│           │                             │                           │        
│           │                             │                           │ random  
│           │                             │                           │ sample  
│           │                             │                           ▼         
│           │                    ┌────────┴────────┐      ┌─────────────────┐  
│           │                    │ Current Q-Net θ │      │  Training Step  │  
│           │                    │(Online Network) │      │                 │  
│           │                    │   Q(s,a;θ)      │      │   每 C 步更新 θ⁻  │  
│           │                    └────────┬────────┘      └─────────┬───────┘  
│           │                             │                         │         
│           │         ┌───────────────────┴───────────────────┐     │         
│           │         │                                       │     │         
│           │         ▼                                       ▼     │         
│           │  ┌───────────────────┐              ┌──────────────────┐
│           ├──►Target Q-Net θ⁻    │              │   Loss = (y-Q)²  │
│           │  │(Fixed Parameters) │              │                  │
│           │  │   Q(s,a;θ⁻)       │              └─────────┬────────┘
│           │  └───────────┬───────┘                        │     
│           │              │                                │ Backprop  
│           │              │ maxₐ'Q(s',a';θ⁻)               │ θ←θ-α∇Loss
│           │              │                                │     
│           │              └─────────────►┌──────────────────┘
│                                          │                    
│                                 y = r + γ·maxₐ'Q(s',a';θ⁻)    
│                                          │                    
│                                          ▼                    
│                                 ┌───────────────────┐         
│                                 │   Target Value    │         
│                                 └───────────────────┘         
└─────────────────────────────────────────────────────────────────────────────
```

#### 【模块作用详解】

#### 【模块作用总结】

| 模块 | 输入 | 输出 | 核心作用 |
|------|------|------|---------|
| **Environment** | 动作 a | (s', r, done) | 提供状态转移和奖励信号 |
| **Policy (ε-Greedy)** | 当前状态 s | 动作 a | ε-探索 + Q 值利用，决定下一个动作 |
| **Replay Buffer** | (s,a,r,s',done) | Batch样本 | 打散相关性、重复利用经验、i.i.d.假设 |
| **Current Q-Net (θ)** | 状态 s | Q(s,·) | 实时预测价值，训练的目标网络 |
| **Target Q-Net (θ⁻)** | 下一状态 s' | maxₐ'Q(s',a';θ⁻) | 固定 target，避免移动靶子问题 |

> **关键设计：** DQN 用"Replay Buffer + Target Network"两个工程技巧解决了理论完美但实践崩溃的问题。

#### 【原理图】Replay Buffer 如何打散相关性

```text
传统 RL（时序训练）：
┌─────────────────────────────────────────────────────┐
│ s₁ → a₁ → r₁ → s₂ → a₂ → r₂ → s₃ ...                │
│ ↑         ↑         ↑                               │
│ └─────────┴─────────┘                               │
│     状态高度相关！                                   │
│     神经网络过拟合局部模式                           │
└─────────────────────────────────────────────────────┘

DQN（Replay Buffer）：
┌─────────────────────────────────────────────────────┐
│ 存储: [(s₁,a₁,r₁,s₂), ..., (sₜ,aₜ,rₜ,sₜ₊₁)]        │
│      ↓                                              │
│ 采样: random_sample(buffer, batch_size)             │
│      → [s₅, s₁₀, s₂, s₈, ...]                       │
│      打散时序相关性 ✅                               │
└─────────────────────────────────────────────────────┘

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

| ✅ 已解决        | ❌ 仍待解决       |
|-----------------|------------------|
| 连续状态空间     | 只能处理离散动作  |
| 自动泛化到新状态 | 样本效率低（百万步） |
| 端到端学习（像素→动作） | 训练不稳定，超参数敏感 |

---

## 第六章预告：Policy Gradient

> **"既然我最终想要的是策略 π(a|s)，能不能别绕道学 Q，直接优化策略本身？"**

这就是 Policy Gradient 要解决的问题。

