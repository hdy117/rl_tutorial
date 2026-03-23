## 第六章：Policy Gradient - 直接优化策略

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

> 你在这里：主干 -> Policy Gradient

### 序：从"先学价值"到"直接学行为"

这一章不要一上来背梯度公式。先抓住那个真正把新方法逼出来的问题：

```text
我最终要优化的是行为策略
但 value-based 方法要先学 Q，再从 Q 间接导出策略
如果动作是连续的，或者我关心的是动作分布本身，这条路就开始变别扭
```

先把这一章的主线钉住：

```text
problem -> starting point -> invention -> verification -> example
```

一句话先压住：

> **Policy Gradient 的本质，是不再把策略当成 value 的附属品，而是直接把策略本身当作优化对象。**

---

### 1. Problem：为什么 value-based 路线会自然逼出 Policy Gradient？

先看前面的方法在干什么：

```text
Q-Learning / DQN:
先学 Q(s,a)
   -> 再选 argmax_a Q(s,a)
   -> 策略是"从 Q 里顺手挑出来的"
```

这条路线在离散动作里很好用，但它天然有三个压力点。

#### 1.1 动作一旦连续，`argmax` 就开始难看

如果动作只是：
- 左 / 右
- 上 / 下

那从 `Q(s,a)` 里挑最大值很容易。

但如果动作变成：
- 方向盘角度
- 关节扭矩
- 油门大小

动作空间是连续的，你就没法再轻松地做一个干净的 `argmax`。为了继续用 value-based，你往往得把连续动作硬离散化，而这会立刻带来：
- 维度爆炸
- 分辨率粗糙
- 控制不自然

#### 1.2 你真正想学的是策略，但系统在逼你先学价值

从任务目标看，agent 最终被执行的不是 `Q`，而是策略 `pi`。

也就是说：
- 真正被部署的是"怎么行动"
- 不是"每个动作值多少钱的中间表征"

这就像你真正想训练的是"驾驶动作"，但当前流程要求你先建立一张"所有可能动作评分表"，再从表里取最大。能用，但绕。

#### 1.3 有时我们关心的就是"动作分布本身"

很多任务里，策略不是一个单点动作，而是一个分布：
- 我想保留随机探索
- 我想输出高斯分布参数
- 我想控制策略熵，让行为不要过早塌缩

这时候"先学值，再取最大"本身就不再是最自然的语言。

所以这一章的原始问题可以压成一句：

> **如果最终目标本来就是策略，那为什么不直接优化策略本身？**

---

### 2. Starting Point：既然目标是行为规则，那就直接表示行为规则

从第一性原理看，出发点其实非常朴素：

> **我要控制什么，就直接把什么写成参数化对象。**

在 RL 里，我们真正想控制的是：

> 在状态 `s` 下，采取动作 `a` 的行为倾向。

那最自然的表示就是：

```math
\pi_\theta(a|s)
```

这里：
- `\theta` 是策略参数
- 输入是状态 `s`
- 输出是一个动作分布或动作概率

如果动作空间离散，它可能输出：

```text
\pi_\theta(a|s) = [0.1, 0.7, 0.2]
```

意思是：
- 动作 1 概率 10%
- 动作 2 概率 70%
- 动作 3 概率 20%

如果动作空间连续，它也可以输出一个分布的参数，比如高斯分布的：
- `mu`
- `sigma`

于是目标函数也跟着改写。

以前我们隐含地在做：
- 先学一个 value
- 再让策略从 value 里长出来

现在我们直接写：

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
```

含义非常直接：
- `\tau` 是按当前策略跑出来的一条轨迹
- `R(\tau)` 是这条轨迹的总回报
- 我要调整 `\theta`，让期望总回报最大

这一步特别重要，因为它完成了目标层面的切换：

```text
旧路线：优化 value，策略是副产物
新路线：直接优化策略，value 不再是必须中介
```

---

### 3. Invention：Policy Gradient 是怎么被"逼出来"的？

一旦你接受"策略本身就是优化对象"，下一个问题就被强行摆到桌面上：

> **参数 `theta` 到底应该朝哪个方向改，才能让策略变好？**

别急着看公式，先看最原始的行为逻辑。

#### 3.1 最朴素的规则：好动作以后更常做，差动作以后少做

如果某次在状态 `s_t` 下选了动作 `a_t`，结果后面拿到了很高的回报，那最自然的更新方向就是：
- 下次在类似状态里，更容易选到这个动作

如果后面回报很差，那最自然的更新方向就是：
- 下次在类似状态里，更不容易选到这个动作

这其实就是 reinforcement 的原型：

```text
好结果 -> 增加这类行为的概率
差结果 -> 降低这类行为的概率
```

#### 3.2 把"增大/减小概率"写成可微的形式

既然策略已经被写成 `\pi_\theta(a|s)`，那"让某个动作更容易被选到"本质上就是：
- 增大 `\pi_\theta(a_t|s_t)`

为了得到可优化形式，我们看它的对数概率：

```math
\log \pi_\theta(a_t|s_t)
```

为什么看 log？因为它在数学上更容易求梯度，也能把概率乘积变成求和，这是后面整条推导能工作的重要压缩方式。

于是，"朝着更可能选到好动作的方向更新"就变成了：

```math
\nabla_\theta \log \pi_\theta(a_t|s_t)
```

它表示：
- 如果我动一下参数 `\theta`
- 这个动作被选中的倾向会怎么变化

#### 3.3 用回报给这个方向加权

但不是每个动作都该一视同仁。动作后果不同，所以更新强度也应该不同。

最直接的做法就是拿这个动作之后的累计回报 `G_t` 来当权重：

```math
\nabla_\theta J(\theta) = \mathbb{E}[G_t \nabla_\theta \log \pi_\theta(a_t|s_t)]
```

这条式子读成人话就是：

- 如果 `G_t` 大，说明这次动作后果好 -> 增大它的概率
- 如果 `G_t` 小，说明这次动作后果差 -> 减小它的概率贡献

于是你会发现，Policy Gradient 根本不是凭空冒出来的技巧，而是下面这条直觉的微分版本：

> **策略参数应该朝着"让高回报动作更常发生"的方向移动。**

#### 3.4 REINFORCE：最纯的基础实现

最基础的 Policy Gradient 算法就是 `REINFORCE`。

它的流程非常朴素：

1. 用当前策略跑完整个 episode
2. 记录每一步 `(s_t, a_t, r_t)`
3. 反向算出每一步之后的累计回报 `G_t`
4. 用 `-log_prob * G_t` 作为 loss 更新参数

伪代码：

```python
for episode in range(num_episodes):
    trajectory = []
    state = env.reset()
    done = False

    while not done:
        action = sample_from(policy_net(state))
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state

    returns = compute_discounted_returns(trajectory)

    loss = 0
    for (state, action, _), G in zip(trajectory, returns):
        log_prob = policy_net.log_prob(state, action)
        loss += -log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这段代码背后的思想只有一句：

> **用最终回报，去重塑产生这些动作的概率分布。**

---

### 4. Verification：怎么验证我们真的发明对了？

一套新方法不是"能写出公式"就算成立，而是要检查它是否真的解决了原问题。

#### 4.1 它有没有直接作用在策略上？

有。

它更新的不是 `Q` 表，也不是某个中间价值映射，而是：

```math
\pi_\theta(a|s)
```

所以目标和被优化对象终于对齐了。

#### 4.2 它能不能自然处理连续动作？

能。

因为策略输出的不必是离散动作标签，它可以直接输出连续分布参数，比如：
- 均值 `mu`
- 方差 `sigma`

然后从这个分布里采样动作。

这意味着：
- 不需要粗暴离散化动作空间
- 不需要在连续动作上做难看的 `argmax`

#### 4.3 它会不会把高回报行为推高、把低回报行为压低？

会。

因为梯度里直接乘了 `G_t`：
- `G_t` 大 -> 更新方向强化这类动作
- `G_t` 小 -> 更新方向削弱这类动作

这刚好符合最初的行为逻辑。

#### 4.4 那为什么它还不够完美？

因为它虽然方向对了，但信号太吵。

`REINFORCE` 用的是整段回报 `G_t`，于是：
- 单次运气波动会很大
- 每一步动作都背着后续整条轨迹的锅
- 学习方差很高，收敛会慢

所以验证的结论是：

```text
Policy Gradient 在"目标对齐"和"连续动作处理"上是对的
但在"学习信号稳定性"上还不够好
```

这就逼出下一章的核心问题：

> **既然直接学策略的方向是对的，那怎么给它一个更稳定的评价器？**

这就是 `Actor-Critic`。

---

### 5. Example：走窄桥的小机器人

这个例子最适合说明"为什么直接学策略是自然的"。

```text
起点 ---- 窄桥 ---- 终点
          |
          +-- 掉下去 = -100
```

机器人每一步要控制：
- 往前迈多少
- 身体偏左还是偏右

这些都不是离散标签，而是连续控制量。

#### 5.1 如果用 Q-table，会发生什么？

你得把动作离散化成很多小格子：
- 偏左 0.01
- 偏左 0.02
- 偏右 0.01
- 偏右 0.02
- ...

很快动作表就爆炸，而且控制会变得很僵硬。

#### 5.2 如果用 Policy Gradient，会发生什么？

策略网络可以直接输出一个连续动作分布，比如：

```text
mu = 0.02
sigma = 0.10
```

意思是：
- 默认平均稍微偏右一点
- 但保留一些探索波动

如果训练后发现：
- 偏右一点更容易走到终点
- 偏左容易掉下桥

那更新后的策略就会逐渐变成：
- `mu` 向安全方向移动
- `sigma` 变小，动作更稳

这正是直接优化行为分布的含义。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
value-based 先学 Q 再导出策略，连续动作和动作分布场景下很绕

Starting Point:
真正目标是策略，那就直接参数化策略 pi_theta(a|s)

Invention:
用回报 G_t 给 log-prob 梯度加权
让高回报动作概率上升，低回报动作概率下降

Verification:
它直接优化策略，天然适合连续动作
但更新信号方差大，不够稳定

Example:
走窄桥机器人直接输出连续动作分布，比查 Q 表更自然
```

### 本章速记卡片

#### 一句话主线

- value-based 是"先学价值，再导出策略"
- Policy Gradient 是"既然目标是策略，那就直接优化策略"
- 它解决了连续动作和策略分布表达问题
- 但它仍然留下高方差问题

#### 必背公式

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
```

```math
\nabla_\theta J(\theta) = \mathbb{E}[G_t \nabla_\theta \log \pi_\theta(a_t|s_t)]
```

#### 必背术语

- `Policy Parameterization`
- `Log-Probability`
- `Return G_t`
- `REINFORCE`

#### 最短背诵版

1. 最终目标是策略，不是 Q 表
2. 所以直接把策略写成 `\pi_\theta(a|s)`
3. 高回报动作提概率，低回报动作降概率
4. 连续动作很友好
5. 但训练信号方差大

---

---

### 7. 核心技术点一：策略参数化详解

#### 7.1 离散动作空间

**数学形式：**
```math
π_θ(a|s) = Softmax(f_θ(s))_a
```

**PyTorch 实现：**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        logits = self.policy_net(state)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def sample_action(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist
    
    def get_log_prob(self, state, action):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)
```

#### 7.2 连续动作空间（高斯策略）

**为什么用 `log_std`？**
- 神经网络输出可能是负数，但标准差必须 > 0
- `std = exp(log_std)` 保证正值
- 梯度更稳定，避免数值溢出

**PyTorch 实现：**
```python
class ContinuousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.log_std_net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        mu = self.mu_net(state)
        log_std = self.log_std_net(state)
        std = torch.exp(log_std)
        return mu, std
    
    def sample_action(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action = torch.tanh(action)  # 映射到 (-1, 1)
        return action, dist
```

#### 📊 图：离散 vs 连续策略输出对比

```text
离散动作空间 (CartPole):
━━━━━━━━━━━━━━━━━━━━━━━
状态 → [神经网络] → logits → Softmax → 概率分布
                              ↓
                    P(左)=0.3, P(右)=0.7
                              ↓ (采样)
                         动作 = 右 ✅

连续动作空间 (Walker2D):
━━━━━━━━━━━━━━━━━━━━━━━
状态 → [神经网络] → μ, σ → N(μ,σ²) → 动作分布
                            ↓
                   N(-0.5, 0.3²), μ=-0.5
                            ↓ (采样 + tanh)
                       动作 = -0.42 ✅
```

---

### 8. 核心技术点二：Policy Gradient Theorem 完整推导

#### 8.1 目标函数的梯度计算

**问题：** `∇_θ J(θ) = ∇_θ E[R(τ)]`，但分布依赖 θ，怎么求？

**技巧——对数导数恒等式：**
对于任何可微函数 f(x) > 0：
```math
∇ f(x) = f(x) * ∇ log f(x)
```

**完整推导步骤：**

1. **原始形式：**
   ```math
   ∇_θ J(θ) = ∫ π_θ(τ) R(τ) dτ
   ```

2. **引入对数技巧：**
   ```math
   ∇_θ J(θ) = ∫ [π_θ(τ) * ∇_θ log π_θ(τ)] R(τ) dτ
   ```

3. **写成期望形式：**
   ```math
   ∇_θ J(θ) = E_{τ~π_θ}[R(τ) * ∇_θ log π_θ(τ)] ✅
   ```

#### 8.2 轨迹概率的分解

```math
π_θ(τ) = p(s₀) * π_θ(a₀|s₀) * p(s₁|s₀,a₀) * π_θ(a₁|s₁) * ...
       = (环境项，与 θ 无关) * ∏_{t=0}^T π_θ(a_t|s_t)
```

取对数后求梯度（常数消失）：
```math
∇_θ log π_θ(τ) = Σ_{t=0}^T ∇_θ log π_θ(a_t|s_t)
```

代回得到实用形式：
```math
∇_θ J(θ) ≈ E[Σ_t G_t * ∇_θ log π_θ(a_t|s_t)]
```

#### 📊 图：对数技巧的直观理解

```text
原始问题：如何求 E[R(τ)] 关于 θ 的梯度？

方法 A (不可行): 
∇_θ Σ π_θ(τ) * R(τ) → 需要知道完整轨迹概率

方法 B (对数技巧):
1. ∇_θ J = ∫ ∇_θ [π_θ(τ)] * R(τ) dτ
2. 用恒等式：∇_θ π_θ(τ) = π_θ(τ) * ∇_θ log π_θ(τ)
3. 代回写成期望形式 ✅

关键优势：∇_θ log π_θ(τ) = Σ_t ∇_θ log π_θ(a_t|s_t)
         → 每一项都是局部的，可计算！
```

---

### 9. 核心技术点三：方差问题与减少技术

#### 9.1 REINFORCE 的方差问题

**更新规则：**
```math
∇_θ J(θ) ≈ G_t * ∇_θ log π_θ(a|s)
```

**方差来源：** `G_t` 包含环境噪声和策略探索噪声

**数值示例：**
```python
# 同一状态 s，同一个动作 a=右，两次采样：

Episode 1: r=[0, -50, +100, ...] → G_t = +45
Episode 2: r=[0, +30, -40, ...]  → G_t = -5

同样的动作，两次回报符号相反！→ 梯度更新可能完全反向
```

#### 📊 图：REINFORCE 方差问题的可视化

```text
REINFORCE 的梯度更新示例 (CartPole):
━━━━━━━━━━━━━━━━━━━━━━━━━━━
第 100 次 episode，状态 s=平衡态：策略输出 P(右)=0.6, 采样到 a=右

Episode A: 后续 +200 分 → G_t = +200 → gradient ∝ +200 * log_prob_grad
Episode B: 后续 -50 分  → G_t = -50  → gradient ∝ -50  * log_prob_grad

结果：同样的动作，有时上调有时下调 → 方差大

实际梯度估计波动 (模拟):
━━━━━━━━━━━━━━━
+300 |    ╭──┐
     │    │  └╮
+150 |    │   ╰─╮
  0  ─┼─╯      └─┼────┼──→ 期望值接近 0，但每次波动极大
     │            │
-150 |    ╭─────╮ │
     │ ╭──┘     ╰─╮
-300 └────────────┴──

对比：带 Baseline 的版本会窄很多！
```

#### 9.2 Baseline 技术：减去常数项

**核心思想：**
如果从 `G_t` 中减去与动作无关的常数 `b(s)`，梯度期望不变但方差降低。

```math
∇_θ J(θ) = E[(G_t - b(s)) * ∇_θ log π_θ(a|s)]
```

**为什么为 0？**
```math
E[b(s) * ∇_θ log π_θ] = b(s) * Σ_a π_θ(a|s) * ∇_θ log π_θ(a|s)
                     = b(s) * ∇_θ Σ_a π_θ(a|s)
                     = b(s) * ∇_θ (1)
                     = 0 ✅
```

**最佳 Baseline：状态价值函数 V(s)**
```math
A(s,a) = G_t - V(s)  ← Advantage Function
```

#### 📊 图：Baseline 如何降低方差

```text
方差对比 (CartPole):
━━━━━━━━━━━━━━━━━━━━━━━
REINFORCE (无 baseline):
+200 |    ╭────╮
     │    │    │
+100 |    │ ╭──┴╮
  0  ─┼─╯      └──→ 期望值 >0，但波动大

REINFORCE + Baseline:
+100 |    ╭──╮
     │    │  │
+50  |    │  │  ╭──╮
  0  ─┼────┴──┴──┤  │→ 期望值接近 0，波动小很多！

方差减少率：~70%（实验数据）
```

#### 9.3 Advantage Function：为什么比 `G_t` 好？

**定义：**
```math
A(s,a) = Q(s,a) - V(s) = E[G_t | s,a] - E[G_t | s]
```

**直观含义：** "在状态 s 下，选动作 a 比平均表现好多少？"

| 方法 | 权重项 | 优点 | 缺点 |
|------|--------|------|------|
| REINFORCE | `G_t` (总回报) | 简单 | 方差大 |
| PG + Baseline | `G_t - b(s)` | 降低方差 | 需要估计 b(s) |
| Actor-Critic | `A(s,a)` | 最优方差减少 | 需要学习 Critic |

---

### 10. 核心技术点四：完整代码示例（CartPole REINFORCE）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        logits = self.net(state)
        return F.log_softmax(logits, dim=-1)

class REINFORCE:
    def __init__(self, policy, lr=1e-2):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    def compute_returns(self, rewards, gamma=0.99):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)
    
    def train_step(self, states, actions, returns):
        log_probs = self.policy(states)
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(-1))
        loss = -(chosen_log_probs * returns).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# 训练主循环
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = PolicyNet(state_dim, action_dim)
agent = REINFORCE(policy, lr=1e-2)

episode_rewards = []
for ep in range(2000):
    state, _ = env.reset()
    states_list, actions_list, rewards_list = [], [], []
    
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        log_prob = policy(state_tensor)
        dist = Categorical(torch.exp(log_prob))
        action = dist.sample()
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        state = next_state
    
    returns = agent.compute_returns(rewards_list, gamma=0.99)
    loss = agent.train_step(
        torch.FloatTensor(np.array(states_list)),
        torch.LongTensor(actions_list),
        returns
    )
    
    episode_rewards.append(sum(rewards_list))
    
    if ep % 100 == 0:
        print(f"Episode {ep}: avg reward = {np.mean(episode_rewards[-100:])}")

# 可视化训练曲线
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.axhline(y=500, color='r', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('REINFORCE on CartPole')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

### 11. 核心技术点五：Actor-Critic 完整实现

#### 11.1 Actor-Critic 架构设计

**核心思想：** 用价值网络估计 `V(s)`，计算 Advantage

```text
┌─────────────────┐
│   State s_t     │ ←─── 输入状态
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    v         v
Actor π_θ   Critic V_w
(策略网络)  (价值网络)
    │         │
    v         v
Action a_t  Value V(s_t)
            ↓
      TD error: r + γ*V(s') - V(s)

训练循环：
- Actor 更新：∇_θ J ≈ A_t * ∇_θ log π_θ(a_t|s_t)
- Critic 更新：∇_w L ≈ (r + γ*V(s') - V(s))²
```

#### 📊 图：Actor-Critic 训练曲线对比（模拟数据）

```text
平均回报随 Episode 的变化:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Episode →      0     200    400    600    800   1000

REINFORCE:   │ ╲│╱│╲│╱│╲│╱│╲│╱│╲│╱│╲│╱│
             │  │  │  │  │  │  │  │  │  │
Actor-Critic:│   ────────╱────────────────
             │           ╱

关键对比:
━━━━━━━━━
收敛速度：REINFORCE ~800 episodes | AC ~300 episodes (快 2.5x)
稳定性：REINFORCE 波动大 (±100) | AC 稳定 (±30)
方差：REINFORCE 高 | AC 低 (Baseline 减少 ~70%)
```

#### 11.2 Actor-Critic PyTorch 实现

```python
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, discrete=True, hidden_dim=128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if discrete:
            self.actor = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)
        else:
            self.actor_mu = nn.Linear(hidden_dim, action_dim)
            self.actor_logstd = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        
        if isinstance(self.actor, nn.Linear):  # 离散
            logits = self.actor(features)
            value = self.critic(features).squeeze(-1)
            return logits, value
        
        else:  # 连续
            mu = self.actor_mu(features)
            log_std = self.actor_logstd(features)
            value = self.critic(features).squeeze(-1)
            return mu, torch.exp(log_std), value
    
    def get_action(self, state, deterministic=False):
        if isinstance(self.actor, nn.Linear):  # 离散
            logits, _ = self.forward(state)
            dist = Categorical(torch.softmax(logits, dim=-1))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.detach(), log_prob.detach()
        
        else:  # 连续
            mu, std, _ = self.forward(state)
            if deterministic:
                action = torch.tanh(mu)
            else:
                dist = Normal(mu, std)
                action = torch.tanh(dist.sample())
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.detach(), log_prob.detach()

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, discrete=True, 
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99):
        self.model = ActorCritic(state_dim, action_dim, discrete)
        self.gamma = gamma
    
    def train_step(self, states, actions, rewards, next_states, dones):
        # Critic Loss (TD error)
        with torch.no_grad():
            _, next_value = self.model(next_states)  # 简化：只返回 value
            
        td_target = rewards + self.gamma * (1 - dones) * next_value
        _, current_value = self.model(states)
        
        critic_loss = nn.functional.mse_loss(current_value, td_target)
        
        # Actor Loss
        advantage = td_target - current_value.detach()
        
        if isinstance(self.model.actor, nn.Linear):  # 离散
            logits, _ = self.model(states)
            dist = Categorical(torch.softmax(logits, dim=-1))
            log_probs = dist.log_prob(actions)
        else:  # 连续
            mu, std, _ = self.model(states)
            dist = Normal(mu, std)
            actions_clamped = torch.clamp(actions, -1, 1)
            log_probs = dist.log_prob(actions_clamped).sum(dim=-1)
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # 更新
        self.actor_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

# 训练示例
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = ActorCriticAgent(state_dim, action_dim, discrete=True)

episode_rewards = []
for ep in range(500):
    state, _ = env.reset()
    
    states_batch, actions_batch, rewards_batch = [], [], []
    
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = agent.model.get_action(state_tensor)
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        states_batch.append(state)
        actions_batch.append(action)
        rewards_batch.append(reward)
        state = next_state
    
    states = torch.FloatTensor(np.array(states_batch))
    actions = torch.LongTensor(actions_batch)
    rewards = torch.FloatTensor(rewards_batch).unsqueeze(-1)
    dones = torch.FloatTensor([1.0])
    
    actor_loss, critic_loss = agent.train_step(states, actions, rewards, 
                                                states, dones)
    
    episode_rewards.append(sum(rewards_batch))
    
    if ep % 50 == 0:
        print(f"Ep {ep}: reward={np.mean(episode_rewards[-50:]):.1f}")

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.axhline(y=500, color='r', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Actor-Critic on CartPole')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

### 12. 核心技术点六：PPO 简介（PG 的进阶版）

#### 12.1 PPO 为什么出现？

**问题：** Actor-Critic 更新步长太大时容易破坏策略。

**解决：** Proximal Policy Optimization (PPO) 引入约束更新步长。

#### 12.2 PPO 的核心公式

```math
L^{CLIP}(θ) = E[min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)]
```

其中：
- `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)`（新旧策略比率）
- `ε` 是裁剪参数（通常 0.1-0.2）

**直观含义：**
- Advantage > 0：限制概率增加不超过 `(1+ε)` 倍
- Advantage < 0：限制概率减少不超过 `(1-ε)` 倍

#### 📊 图：PPO 的裁剪机制

```text
PPO Clip 的直观理解:
━━━━━━━━━━━━━━━━━━━━━━━

Advantage > 0 (好动作):
旧概率 π_old = 0.3
新概率 π_new = ?

如果 π_new / π_old ≤ 1.2:
  loss = r * A ✅ (正常更新)

如果 π_new / π_old > 1.2:
  loss = 1.2 * A ✂️ (裁剪掉，不再优化)

Advantage < 0 (坏动作):
如果 π_new / π_old ≥ 0.8:
  loss = r * A ✅

如果 π_new / π_old < 0.8:
  loss = 0.8 * A ✂️ (限制下降速度)

结果：策略更新被"约束"在合理范围内，更稳定!
```

#### 12.3 PPO PyTorch 简化实现

```python
class PPOLoss:
    def __init__(self, eps_clip=0.2):
        self.eps_clip = eps_clip
    
    def compute(self, old_log_probs, new_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        
        surr1 = ratio * advantages
        
        ratio_clipped = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip)
        surr2 = ratio_clipped * advantages
        
        loss = -torch.min(surr1, surr2).mean()
        return loss

class PPO:
    def __init__(self, model, lr=3e-4, eps_clip=0.2):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = PPOLoss(eps_clip)
    
    def train(self, old_data, new_states, advantages):
        old_states, old_actions, old_log_probs, _ = old_data
        
        logits, values = self.model(new_states)
        
        if isinstance(self.model.actor, nn.Linear):  # 离散
            new_dist = Categorical(torch.softmax(logits, dim=-1))
            new_log_probs = new_dist.log_prob(old_actions)
        else:  # 连续
            mu, std, _ = self.model(new_states)
            new_dist = Normal(mu, std)
            actions_clamped = torch.clamp(old_actions, -1, 1)
            new_log_probs = new_dist.log_prob(actions_clamped).sum(dim=-1)
        
        loss = self.loss_fn.compute(old_log_probs, new_log_probs, advantages)
        value_loss = nn.functional.mse_loss(values, old_values)
        entropy = new_dist.entropy().mean()
        
        total_loss = loss + 0.5 * value_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()
```

---

### 13. 本章核心技术总结表

| 技术 | 核心公式/方法 | 解决的问题 | 优点 | 缺点 |
|------|--------------|-----------|------|------|
| **策略参数化** | π_θ(a\|s) | 连续动作表达 | 天然支持连续空间 | 需要采样，方差大 |
| **对数技巧** | ∇logπ * R | 梯度估计 | 无偏，可微 | 高方差 |
| **REINFORCE** | G_t * ∇logπ | 基础 PG 实现 | 简单易懂 | 方差极大，收敛慢 |
| **Baseline** | (G_t - b(s)) * ∇logπ | 降低方差 | 简单有效 | 需要选合适的 b |
| **Advantage** | A(s,a) = G_t - V(s) | 信用分配 | 最优方差减少 | 需要 Critic |
| **Actor-Critic** | TD error 作为 Advantage | 稳定性 + 效率 | 样本效率高 | 双网络训练复杂 |
| **PPO** | Clip loss + 约束步长 | 避免策略破坏 | 稳定、采样高效 | 超参数敏感 |

---

### 14. 视觉总结：Policy Gradient 技术演进树

```text
Policy Gradient (直接优化策略)
│
├─ REINFORCE (Monte Carlo PG)
│  ├─ 核心：∇J ≈ G_t * ∇logπ(a|s)
│  └─ 问题：方差极大 ❌
│
├─ Baseline Reduction
│  ├─ 方法：(G_t - b(s)) * ∇logπ
│  └─ 关键:b(s)=V(s)最优
│
├─ Actor-Critic (AC)
│  ├─ Actor: π_θ(a|s) ← 策略网络
│  ├─ Critic: V_w(s) ← 价值网络
│  └─ Advantage: A = r + γ*V(s') - V(s) (TD error)
│
└─ PPO (Proximal Policy Optimization)
   ├─ Clip Loss: min(r*A, clip(r)*A)
   ├─ Constraint: |π_new/π_old| ∈ [0.8, 1.2]
   └─ Why: 避免单次更新破坏策略 ✅

引出下一章:SAC (Soft Actor-Critic)
→ 最大熵 RL，连续控制更优
```

---

### 15. 学习检查清单

#### 概念理解（能回答才算懂）

- [ ] **为什么 Value-Based 方法不适合连续动作空间？**
  - 答案提示：argmax 需要离散枚举，无法直接作用于连续参数
  
- [ ] **对数技巧的核心恒等式是什么？为什么用 log？**
  - 答案提示：∇f = f * ∇logf；数值稳定 + 梯度计算方便
  
- [ ]**G_t 和 Advantage 有什么区别？**
  - 答案提示：A = G_t - V(s)，相对优势 vs 绝对回报
  
- [ ] **为什么减去 Baseline 不改变期望但降低方差？**
  - 答案提示：E[∇logπ]=0，所以 E[b*∇logπ]=0

#### 技术对比

- [ ] REINFORCE vs Actor-Critic 的训练曲线差异
- [ ] 离散动作（Softmax）vs 连续动作（高斯采样）的表示区别
- [ ] PPO 为什么比 AC 更稳定？

#### 代码实现能力

- [ ] 能写出离散动作策略网络的 forward()
- [ ] 能写出高斯采样的 sample + log_prob 计算
- [ ] 能用 PyTorch 实现 REINFORCE 的一个训练步
- [ ] 能理解并解释 PPO 的 clip loss 公式

---

### 16. 下一章预告：Actor-Critic → DDPG/SAC

**核心问题：** 既然 AC 已经降低了方差，为什么还需要 SAC？

**答案要点：**
1. AC + TD3/DDPG = 确定性策略梯度（适合连续控制）
2. SAC = 最大熵 RL（探索更强，性能更好）
3. Off-policy vs On-policy 的效率差异

---

### 17. 核心代码文件清单

建议保存以下文件：

```
rl_tutorial/chapters/
├── chapter_06_Policy_Gradient.md (本文档)
├── code/
│   ├── policy_networks.py        # 离散 + 连续策略网络
│   ├── reinforce.py              # REINFORCE 完整实现
│   ├── actor_critic.py           # AC 实现
│   └── ppo_simplified.py         # PPO 简化版
├── plots/
│   ├── pg_family_tree.png        # PG 技术演进树
│   ├── variance_comparison.png   # REINFORCE vs AC 方差对比
│   └── gaussian_evolution.gif    # 高斯策略 μ,σ演化动画
└── exercises/
    ├── exercise_6_1.py           # 离散动作 CartPole PG
    ├── exercise_6_2.py           # 连续控制 Walker2D AC
    └── exercise_6_3.py           # PPO from scratch
```

---

## 🎯 现在该做什么？

1. **代码实践**：把上面的 REINFORCE + Actor-Critic 代码跑起来，看训练曲线差异
2. **可视化**：画出方差对比图、策略分布演化动画
3. **习题**：自己实现一个 PPO(简化版)，在 Ant-v4 上跑一下

下一章我们继续深入：**SAC (Soft Actor-Critic)** —— 最大熵 RL，连续控制的最优选择 🔥

