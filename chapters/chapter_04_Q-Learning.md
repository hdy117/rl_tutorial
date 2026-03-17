## 第四章：Q-Learning（让 Agent 开始真正学习）

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

> 你在这里：主干 → Q-Learning

---

### 序：为什么 Bellman 方程还不够？

第三章里我们得到了两条方程：

- `Bellman Expectation Equation` — 评估当前策略有多好
- `Bellman Optimality Equation` — 告诉我最优动作价值应该满足什么关系

但这里有个**致命问题**：

> **我知道正确答案该长什么样，可我该怎么在不知道环境模型的情况下，把它学出来？**

这就是第四章要接手的问题。

---

### 🔥 独立重构练习（先别往下读）

藏起公式，自己推导：

1. **如果我想选动作，只知道 `V(s)` 够不够？为什么必须需要 `Q(s,a)`？**
2. **`Bellman Optimality Equation` 里的期望怎么在现实中计算？有什么可用信息源？**
3. **更新公式应该长什么样？从"旧估计 + 修正量"的角度推导**

5 分钟后再看下面的答案。

---

## 1. 什么问题逼出了 Q-Learning？

第三章的 Bellman 方程告诉我们：

```math
Q^*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a')]
```

但这个式子有个**无法执行的问题**：

- `E` 表示期望，要对所有可能的未来轨迹求平均
- 可现实里我们**不知道环境模型**（转移概率、奖励分布）
- 所以没法精确计算右边的期望

那怎么办？

> **唯一可用的信息源是真实交互得到的 sample。**

这就是 Q-Learning 被迫要解决的问题：

> **在没有完整环境模型的情况下，怎么用单次样本逼近 Bellman Optimality？**

---

## 2. 为什么 `V(s)` 不够，必须学 `Q(s,a)`？

假设我只有状态价值函数 `V(s)`。它能告诉我：

- "这个状态整体值多少钱"

但真正做决策时，我需要的是：

- **往左好还是往右好？**
- **现在加速值不值？**
- **哪个动作会把我带向更好的未来？**

`V(s)` 无法回答这些问题 — 因为它是对**所有可能动作的平均**（或最优），但不区分具体动作。

所以这逼出一个更细的量：

```math
Q(s,a)
```

它表示：

> **在状态 `s` 下，如果我先做动作 `a`，未来总回报大概是多少？**

一句话：

> **`V(s)` 告诉你状态好不好，`Q(s,a)` 告诉你在该状态下哪个动作更值。**

---

## 3. Q-Learning 的表示方式：一张表就够了吗？

既然目标是学出每个 `(s, a)` 的价值，最自然的表示就是一张 `Q-table`：

```text
状态 s × 动作 a → Q 值
```

在离散环境里它可能长这样：

```text
          ←       →
A       0.12    0.45
B       0.30    0.81
C       0.10    0.90
```

这张表一旦学准，策略就不需要额外发明了：

```math
\pi(s) = \arg\max_a Q(s,a)
```

**关键洞察：**

- Q-table 不是规则表（不写"该怎么做"）
- Q-table 是动作价值表（只记录"每个动作值多少"）
- **策略会从价值里自然长出来，而不是被硬编码**

---

## 🔥 探索 vs 利用：为什么这是被迫的？

现在有个新问题浮现了。

假设我初始化 Q-table 为全零，然后永远选当前最优动作 `argmax(Q)`。会发生什么？

- 初始时所有 `Q(s,a) = 0`
- 第一次在某状态随机（或默认）选了某个动作
- 如果这个动作运气好拿到正 reward → Q 值上升
- 以后在这个状态永远选它
- **结果：其他可能更好的动作永远没机会被尝试**

这就是**局部最优陷阱**。

所以这逼出一个结论：

> **Q-Learning 必须引入随机性，否则初始状态的偶然选择会锁死策略。**

这就叫 **Exploration（探索）** vs **Exploitation（利用）**。

最经典的解决方案是 `epsilon-greedy`：

- 以概率 `ε` 随机选动作（探索）
- 以概率 `1-ε` 选当前最优动作（利用）

训练时通常让 `ε` 逐渐下降：前期多探索，后期多利用。

> **这是被迫的：因为初始 Q 值是零/随机，如果不强制探索，策略会被偶然性锁死。**

---

## 4. 如何从 Bellman Optimality 推导出 Q-Learning？

回到 `Bellman Optimality Equation`：

```math
Q^*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a')]
```

这个式子告诉我们的不是算法步骤，而是：

> **正确的最优动作价值，应该满足什么结构。**

用人话翻译：

- 当前动作价值 = 当前 reward + 下一状态中最好的未来价值

但这里有个**工程障碍**：

- 方程里有 `E`（期望）
- 现实里没有完整环境模型
- 没法精确计算右边的期望

那怎么办？

> **用一次真实交互得到的 sample，去近似这个期望。**

也就是：

1. 执行动作 `a`，得到真实的 `(r, s')`
2. 用当前 Q-table 估计 `s'` 里的最优未来价值 `max_{a'} Q(s', a')`
3. 拼成 Bellman target: `r + γ * max_{a'} Q(s', a')`
4. 把旧的 `Q(s,a)` 往这个 target 推近一点

于是得到：

```math
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]
```

**验证这个推导：**

- ✅ 保留 Bellman 结构？是，仍然是 `当前 reward + 最优未来价值`
- ✅ 保留 optimality 目标？是，未来项仍是 `max`
- ✅ 把不可计算的期望变成可执行步骤？是，用单次 sample 近似环境期望

一句话：

> **Q-Learning = Bellman Optimality Equation 的样本版本。**

---

## 5. Q-Learning Update 公式拆解

```math
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'}Q(s',a') - Q(s,a)]
```

### 第 1 部分：`Q(s,a)`（旧估计）

- 我之前认为，在状态 `s` 做动作 `a` 值这么多

### 第 2 部分：`r + γ * max Q(s', a')`（Bellman Target）

- **眼前收益**：这一步立刻拿到的 reward
- **未来收益**：下一状态里最好的动作价值
- target = 眼前 + 未来

### 第 3 部分：`target - Q(s,a)`（TD Error）

- **新证据和旧估计的差距**
- 正数 → 原来估低了，该往上修
- 负数 → 原来估高了，该往下压

### 第 4 部分：`α * (...)`（学习率）

- 控制每次修正的力度
- `α` 大 → 改得更猛（可能震荡）
- `α` 小 → 改得更稳（收敛慢）

---

## 6. 一个极小示例：Q 值更新是怎么发生的？

假设当前在状态 `B`，执行动作 `→`：

| 参数 | 值 |
|------|-----|
| reward `r` | 0 |
| 下一状态 | `C` |
| `max_{a'} Q(C, a')` | 0.90 |
| `gamma (γ)` | 0.9 |
| `alpha (α)` | 0.1 |
| `Q(B, →)`（旧值） | 0.40 |

**Step 1：计算 target**
```math
target = r + γ × max_{a'} Q(s', a') = 0 + 0.9 × 0.90 = 0.81
```

**Step 2：计算 TD error**
```math
TD error = target - old_Q = 0.81 - 0.40 = 0.41
```

**Step 3：更新 Q 值**
```math
Q(B, →) ← 0.40 + 0.1 × 0.41 = 0.441
```

**含义：**

- 原来我以为 `B →` 值 `0.40`
- 新经验告诉我它应该接近 `0.81`
- 所以往新证据修正一点点，变成 `0.441`

一次更新只挪一点，但很多次之后，整张 Q-table 就会越来越准。

---

## 7. Q-Learning 的完整训练流程

```text
初始化 Q-table（全零）
    ↓
环境 reset → 得到初始状态 s
    ↓
epsilon-greedy 选动作 a
    ↓
执行动作 → 得到 (r, s')
    ↓
用 update 公式修正 Q(s,a)
    ↓
状态变成 s'
    ↓
如果没结束，继续循环
    ↓
episode 结束 → 开始下一局
```

**两个核心效果：**

1. **奖励逐步向前传播** — 终点的高 reward 会像涟漪一样往前扩散
2. **策略从 Q-table 自然长出来** — 不需要手写规则，只需选 `argmax(Q)`

---

## 8. 对应到代码：`03_q_learning.py` 在做什么？

### 8.1 建环境

```python
env = gym.make("CartPole-v1")
```

- 任务是 CartPole（小车杆不倒）

### 8.2 离散化连续状态

CartPole 的观察是连续的：
- 小车位置 `0.137`
- 杆角度 `-0.024`

但 Q-table 需要离散索引，所以要 discretize：

```python
def discretize(state):
    # 把连续值压到 NUM_BINS 个桶里
    return (bin1, bin2, bin3, bin4)
```

### 8.3 初始化 Q-table

```python
q_table = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, action_size))
```

- 初始全零，表示"我对所有状态动作都没有经验"

### 8.4 Epsilon-Greedy 选动作

```python
def choose_action(state, epsilon):
    if random() < epsilon:
        return random_action()  # 探索
    else:
        return argmax(q_table[state])  # 利用
```

- 前期多随机试错，后期多相信自己学到的东西

### 8.5 Q-Learning Update

```python
old_q = q_table[state][action]
next_max = np.max(q_table[next_state])
target = reward + GAMMA * next_max
q_table[state][action] = old_q + ALPHA * (target - old_q)
```

- 和公式一一对应：`Q ← Q + α(target - Q)`

### 8.6 Epsilon Decay

```python
epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
```

- 开始多探索，逐渐减少随机性

---

## 🔥 独立重构练习（藏起答案再试一次）

现在盖住公式，自己从头推导 Q-Learning：

1. **为什么 `V(s)` 不够？必须学什么？** → `Q(s,a)`
2. **Bellman Optimality 里的期望怎么计算？** → 用 sample 近似
3. **更新方向应该是什么？** → 向 Bellman target 靠近
4. **为什么需要 epsilon-greedy？** → 初始 Q 值随机，必须强制探索

如果每一步都能独立重建，说明你真的理解了。

---

## 9. Q-Learning 的局限与下一步

### ✅ 优点

- 思想清晰：从 Bellman 直接推导出来
- 代码简单：几行就能实现
- 适合入门：把 RL 核心概念串起来

### ❌ 局限

- **Q-table 需要离散状态** — 连续空间要 discretize，信息会损失
- **状态爆炸** — 状态维度一多，表就存不下
- **泛化能力弱** — 没见过的状态完全不会处理

这就是为什么后面要走 DQN：

> **用神经网络代替 Q-table，让模型学会"没见过的状态该怎么估计价值"。**

---

## 本章主线回顾

```text
问题：Bellman 方程告诉我正确答案的结构，但怎么学出来？
↓
出发点：V(s) 不够选动作，必须学 Q(s,a)
↓
表示方式：Q-table 记录每个 (s,a) 的价值
↓
探索困境：初始 Q 值随机，不强制探索会被锁死 → epsilon-greedy
↓
更新规则：用 sample 逼近 Bellman Optimality
↓
验证：保留结构、保留目标、可执行
```

一句话收束：

> **Q-Learning = 在没有环境模型的情况下，用样本数据一步步逼近 Bellman Optimality。**

---

## 本章速记卡片

### 🔑 核心洞察

1. `V(s)` 只告诉你状态好不好，`Q(s,a)` 直接告诉你在该状态下哪个动作更值
2. Q-Learning 学的是一张 Q-table，策略从价值里自然长出来
3. **初始 Q 值为零 → 必须强制探索（epsilon-greedy），否则会被偶然性锁死**
4. Update target = `当前 reward + γ * 下一状态最优未来`
5. TD error = target - old_Q，告诉你是该往上调还是往下压

### 📐 必背公式

```math
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'}Q(s',a') - Q(s,a)]
```

- `α`：学习率（每次修正的力度）
- `[...]`：TD error（新证据和旧估计的差距）

### 🎯 最短背诵版

1. `Q` 比 `V` 更适合直接选动作
2. Q-Learning = Bellman Optimality + Sample Update
3. Epsilon-Greedy 是**被迫的**：初始 Q 值随机，不探索会锁死
4. TD error 驱动更新 → 向更合理的目标靠近
5. 状态太大时 Q-table 不够用 → DQN 出场

---

*下一章：DQN — 当状态空间爆炸，神经网络如何接管 Q-Learning？*
