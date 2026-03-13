# Reinforcement Learning 入门教程

> 视觉学习者友好版 🔥  
> 混合中英文，图多字少

---

## 第一章：RL 是什么？

### 核心循环

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   AGENT  ──── action ────▶  ENVIRONMENT         │
│     ▲                           │               │
│     │                           │               │
│     └────── reward + state ─────┘               │
│                                                 │
└─────────────────────────────────────────────────┘
```

Agent 活在一个循环里：
1. **See** — 观察当前状态 (state)
2. **Do** — 执行动作 (action)
3. **Get** — 收到奖励 + 新状态 (reward + new state)
4. **Repeat** — 不断循环，慢慢变聪明

---

### 具体例子：格子迷宫 🐭

```
┌───┬───┬───┬───┐
│ 🐭 │   │   │   │
├───┼───┼───┼───┤
│   │ ■ │   │   │   ■ = 墙（不能走）
├───┼───┼───┼───┤
│   │   │   │ 🧀 │   🧀 = 奖励 (+1)
└───┴───┴───┴───┘
```

| 概念 | 定义 | 迷宫例子 |
|---|---|---|
| **State (s)** | Agent 当前观察到的情况 | 鼠标所在格子 (row, col) |
| **Action (a)** | Agent 可以执行的操作 | ↑ ↓ ← → |
| **Reward (r)** | 环境给的反馈信号 | 到达🧀得 +1，其他为 0 |

---

### 为什么用 RL 而不是写规则？

```
传统编程：                    强化学习：

规则 ──▶ 电脑 ──▶ 输出        经验 ──▶ Agent ──▶ 规则
```

RL 适合**规则难以手写**的场景 — 机器人走路、下棋、打游戏。

---

## 第二章：Policy（策略）π

### 什么是 Policy？

Policy 就是 agent 的**决策手册**：给定状态，输出动作。

```
State ──▶ Policy (π) ──▶ Action

  🐭 at (0,0)  ──▶  π  ──▶  →
  🐭 at (0,1)  ──▶  π  ──▶  ↓
  🐭 at (1,0)  ──▶  π  ──▶  →
```

希腊字母 **π (pi)** = policy，RL 论文里随处可见。

---

### 两种 Policy

```
┌─────────────────────────────────────────────────────┐
│  DETERMINISTIC（确定性）  │  STOCHASTIC（随机性）     │
│                           │                          │
│  状态 → 唯一动作           │  状态 → 动作概率分布       │
│                           │                          │
│  (0,0) → 永远走 →         │  (0,0) → 70% →           │
│                           │          20% ↓           │
│  像 GPS 导航               │          10% ↑           │
│                           │                          │
│                           │  像考试乱蒙的学生 😅       │
└─────────────────────────────────────────────────────┘
```

- 训练早期 → **Stochastic**：随机探索，啥都试
- 训练成熟 → **Deterministic**：自信果断，直奔目标

---

### RL 的终极目标（一句话）

> **找到策略 π，使得累计 reward 最大化。**

其他所有算法，都是在解决"怎么找到这个 π"的问题。

---

## 第三章：Value Function 与 Bellman 方程

### 1. 为什么需要 Value Function？

先看一个问题：**只靠即时 reward，Agent 能学会长期决策吗？**

```
只看眼前 Reward:        看长远 Value:

┌───┬───┬───┐          ┌────┬────┬────┐
│ 0 │ 0 │ 0 │          │0.73│0.81│0.90│
├───┼───┼───┤    vs    ├────┼────┼────┤
│ 0 │ 0 │ 0 │          │0.50│ ■  │1.00│
├───┼───┼───┤          ├────┼────┼────┤
│ 0 │ 0 │ 🧀│          │0.30│0.50│ 🧀 │
└───┴───┴───┘          └────┴────┴────┘

只看眼前 -> 全是0，懵逼    看长远 -> 知道往哪走 ✅
```

问题在于：

- 很多环境里的 **reward 是稀疏的**
- 没到终点之前，agent 看到的 reward 可能一直都是 `0`
- 如果只盯着眼前 reward，agent 不知道自己是不是在接近目标

所以我们需要一个更强的概念：

> **Value Function = 从现在这个位置出发，未来总共大概能拿到多少 reward。**

一句话区分：

- **Reward**: 这一步立刻拿到了多少
- **Value**: 从这里开始，未来整体值多少

---

### 2. Return 与折扣因子 gamma

Value 之所以有意义，是因为 RL 不只关心当前一步，而是关心 **未来总回报**。

这个未来总回报叫做 **Return**:

```math
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
```

时间线：

```
现在      +1步      +2步      +3步
 r1    +   r2    +   r3    +   r4
      折扣后变成：
 r1    + gamma*r2 + gamma^2*r3 + gamma^3*r4
```

这里的 `gamma` 是 **折扣因子**，取值在 `0 ~ 1` 之间。

- `gamma = 0.9` -> 很在乎未来，目光长远
- `gamma = 0.1` -> 更在乎眼前，未来影响很小

**为什么要打折？**

因为通常我们认为：
- 越远的未来越不确定
- 越晚拿到的奖励，价值应该稍微低一点

举个例子：

```
路径： (0,0) -> (0,1) -> (0,2) -> 🧀
奖励：   0        0        0      +1
gamma = 0.9
```

那么：

```math
V(0,0) = 0.9^3 \\times 1 = 0.729
```

如果你绕远路，多走一步才吃到奶酪：

```math
V(弯路) = 0.9^4 \\times 1 = 0.656
```

所以 agent 会自然偏向更短的路。

---

### 3. 两类 Value：V(s) 和 Q(s, a)

Value Function 常见有两种。

#### 3.1 状态价值函数 `V^pi(s)`

```math
V^\pi(s)
```

意思是：

> **在策略 `pi` 下，从状态 `s` 出发，未来期望总回报是多少。**

重点：它是“这个状态整体值多少”。

---

#### 3.2 动作价值函数 `Q^pi(s, a)`

```math
Q^\pi(s, a)
```

意思是：

> **在策略 `pi` 下，在状态 `s` 先做动作 `a`，然后继续按 `pi` 行动时，未来期望总回报是多少。**

重点：它不仅看状态，还看“在这个状态下选哪个动作”。

---

#### 3.3 二者的区别

```
┌──────────────────────────────────────────────────────┐
│  V(s)                      │  Q(s, a)                │
│                            │                         │
│  "这个状态值多少？"           │  "这个状态下，这个动作值多少？" │
│                            │                         │
│  只看状态                   │  看状态 + 动作            │
│                            │                         │
│  V(🐭在(0,0)) = 0.73       │ Q(🐭在(0,0), →) = 0.73  │
│                            │ Q(🐭在(0,0), ↓) = 0.50  │
│                            │ Q(🐭在(0,0), ←) = 0.10  │
└──────────────────────────────────────────────────────┘
```

所以：

- `V(s)` 更像是“这个位置好不好”
- `Q(s,a)` 更像是“在这个位置做这个动作好不好”

**Q 更适合直接做决策**，因为它可以直接比较动作。

---

#### 3.4 `pi` 是什么意思？

`pi` 就是 **Policy（策略）**。

所以：

```math
V^\pi(s)
```

不是说“状态 `s` 天生有一个固定价值”，而是说：

> **如果你以后都按策略 `pi` 行动，那么状态 `s` 值多少？**

同一个状态，在不同策略下，value 可能完全不同。

```
同一个状态 s

聪明策略 pi1 -> 快速接近目标 -> V^pi1(s) 较高
笨策略   pi2 -> 到处乱走     -> V^pi2(s) 较低
```

同理：

- `V^pi(s)` / `Q^pi(s,a)` -> 某个具体策略下的价值
- `V*(s)` / `Q*(s,a)` -> 最优策略下的价值

---

### 4. Bellman 的核心思想

Bellman 方程之所以重要，是因为它抓住了 RL 里一个最核心的递推关系：

> **长期价值 = 当前收益 + 剩余未来价值**

先从 return 的定义出发：

```math
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
```

把后面的部分提出来：

```math
G_t = R_{t+1} + \gamma G_{t+1}
```

这一步非常关键。

它告诉我们：
- 一整个很长的未来
- 可以拆成“眼前一步”
- 再加上“下一步开始的未来”

这就是 **Bellman 思想** 的来源。

一句话总结：

> **Bellman 的本质，就是把一个长远问题拆成一个递推问题。**

---

### 5. Bellman Expectation Equation

如果我们已经固定了一套策略 `pi`，那么 value 应该满足：

```math
V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s]
```

这叫做 **Bellman expectation equation**。

它表达的是：

> **在策略 `pi` 下，一个状态的价值 = 这一步的即时 reward + 下一状态价值的折扣期望。**

如果写成动作价值函数的形式：

```math
Q^\pi(s,a) = \mathbb{E}_\pi [R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) \mid S_t=s, A_t=a]
```

这一类 equation 的关键词是：

- **expectation**
- **给定 policy**
- **按当前策略继续往后走**

所以它回答的问题是：

> **如果以后继续按这套策略行动，那么现在值多少？**

---

### 6. Bellman Optimality Equation

如果我们不想评估“某个固定策略有多好”，而是想知道“理论上最优能有多好”，就会得到 **Bellman optimality equation**。

对于最优状态价值：

```math
V^*(s) = \max_a \sum_{s',r} p(s', r \mid s,a) [r + \gamma V^*(s')]
```

对于最优动作价值：

```math
Q^*(s,a) = \sum_{s',r} p(s', r \mid s,a) [r + \gamma \max_{a'} Q^*(s',a')]
```

它的核心变化只有一个：

- expectation：未来继续按当前 policy 走
- optimality：未来每一步都取最优动作

一句话理解：

> **Bellman expectation 是“这套策略有多值”，Bellman optimality 是“最优情况下能多值”。**

而 Q-Learning 用的，正是这个 optimality 思想。

---

### 7. Bellman Equation vs Bellman Update

这两个名字很像，但不是一回事。

#### 7.1 Bellman Equation

Bellman equation 是一个 **数学关系**，描述“正确的 value 应该满足什么条件”。

例如：

```math
V(s) = r + \gamma V(s')
```

它更像是：

- 理想目标
- 正确答案应该满足的关系

---

#### 7.2 Bellman Update

但训练时，我们通常并不知道真正的 value。
所以实际做法不是“直接写出答案”，而是**一点点逼近它**：

```math
V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]
```

含义是：

- 旧估计：`V(s)`
- 新目标：`r + gamma * V(s')`
- 用学习率 `alpha` 朝目标走一步

也就是：

```text
新值 = 旧值 + 学习率 * (目标值 - 旧值)
```

所以：

- **Bellman equation** = 定义“对”是什么
- **Bellman update** = 定义“怎么往对的方向逼近”

---

### 8. TD Error：更新时到底在修什么？

在 Bellman update 里，中括号这一项：

```math
r + \gamma V(s') - V(s)
```

叫做 **TD error（Temporal Difference Error）**。

记作：

```math
\delta = r + \gamma V(s') - V(s)
```

它的含义非常直白：

> **我原来对这个状态的估计，和这次新看到的证据相比，差了多少。**

- 如果 `delta > 0` -> 原来估低了，往上调
- 如果 `delta < 0` -> 原来估高了，往下调

举个小例子：

- 当前 `V(B)=0.20`
- 下一状态 `V(C)=0.90`
- `r=0`
- `gamma=0.9`
- `alpha=0.5`

那么：

```math
target = 0 + 0.9 	imes 0.90 = 0.81
```

```math
\delta = 0.81 - 0.20 = 0.61
```

更新后：

```math
V(B) \leftarrow 0.20 + 0.5 	imes 0.61 = 0.505
```

也就是说：

- 原来估计太低：`0.20`
- 更合理目标是：`0.81`
- 这次先向目标靠近一半，变成 `0.505`

---

### 9. Bellman 如何把奖励向前传播

Bellman 最直观的图像是：**奖励像热量一样向前传导。**

看一个最简单的一维世界：

```
A  ->  B  ->  C  ->  🧀
```

规则：
- 到奶酪得到 `+1`
- 其他位置 reward 都是 `0`
- `gamma = 0.9`

初始：

```
V(A)=0.00, V(B)=0.00, V(C)=0.00, V(🧀)=1.00
```

第一轮：

```math
V(C) = 0 + 0.9 	imes 1.00 = 0.90
```

第二轮：

```math
V(B) = 0 + 0.9 	imes 0.90 = 0.81
```

第三轮：

```math
V(A) = 0 + 0.9 	imes 0.81 = 0.729
```

于是形成 value 梯度：

```
0.729  ->  0.81  ->  0.90  ->  1.00
```

agent 即使看不到终点，也能顺着 value 变高的方向前进。

这就是 Bellman 方程最有力量的地方：

> **它让终点的奖励可以一层层向前传播，最终照亮整张地图。**

---

### 10. 为什么 Bellman 更新通常会收敛？

初学时不需要背完整证明，只要抓住直觉：

- 每次更新都在朝 Bellman target 靠近
- 未来 reward 前面会乘 `gamma`
- 当 `gamma < 1` 时，越远的影响越弱
- 所以误差不会无限放大，通常会逐渐稳定下来

一句话版：

> **奖励往前传播，但每传播一层都会打折，所以系统通常会越来越稳定，而不是爆炸。**

---

### 11. Bellman 在 Q-Learning 里怎么落地

Q-Learning 学的不是 `V(s)`，而是：

```math
Q(s,a)
```

它使用的更新公式是：

```math
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'}Q(s',a') - Q(s,a)]
```

把它翻译成人话：

1. 看当前这一步拿了多少 reward
2. 看下一状态里“最好的动作”值多少
3. 把两者合起来，得到一个 Bellman target
4. 用这个 target 去修正当前的 Q 值

对应到代码里通常就是：

```python
target = reward + GAMMA * np.max(q_table[next_state])
old_q = q_table[state][action]
q_table[state][action] = old_q + ALPHA * (target - old_q)
```

逐项对应：

- `reward` -> 当前即时反馈
- `np.max(q_table[next_state])` -> 下一状态最好的未来价值
- `reward + gamma * max(...)` -> Bellman target
- `old_q + alpha * (target - old_q)` -> 朝目标修正一步

所以 Q-Learning 的本质就是：

> **不断用 Bellman optimality 的思想修正 Q 表，最后把整张动作价值地图学出来。**

---

### 12. 这一章的主线回顾

这一章真正想讲清楚的只有一条线：

1. **Reward 不够，需要 Value**
2. **Value 是对未来总回报的估计**
3. **Value 分成 `V^pi(s)` 和 `Q^pi(s,a)`**
4. **Bellman 把长期价值拆成“当前收益 + 未来价值”**
5. **给定策略时，用 Bellman expectation equation 描述 value**
6. **追求最优时，用 Bellman optimality equation 描述最优 value**
7. **训练时不能直接得到真值，只能用 update 一步步逼近**
8. **Q-Learning 就是在用这种方式学习 Q 表**

一句话收尾：

> **Value Function 是要学的地图，Bellman 方程是地图的递推规则，Q-Learning 是把地图学出来的方法。**

---


## 第三章速记版（考试 / 复习快速回顾）

### 一句话主线

- `Reward` 看眼前，`Value` 看长远
- `Value Function` 是对未来总回报的估计
- `Bellman` 把长期价值拆成“当前收益 + 未来价值”
- `Q-Learning` 用这个递推关系一步步学出 Q 表

---

### 1. 必背概念

- **Reward**：当前这一步立刻拿到多少奖励
- **Return**：未来奖励的折扣和
- **Value**：从当前状态出发，未来整体值多少
- **Policy `pi`**：Agent 采取动作的规则
- **`gamma`**：折扣因子，越大越看重未来

---

### 2. 必背公式

#### Return

```math
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
```

#### Return 的递推形式

```math
G_t = R_{t+1} + \gamma G_{t+1}
```

这就是 Bellman 思想的来源。

#### 状态价值函数

```math
V^\pi(s)
```

表示：在策略 `pi` 下，状态 `s` 的长期价值。

#### 动作价值函数

```math
Q^\pi(s,a)
```

表示：在策略 `pi` 下，在状态 `s` 先做动作 `a` 的长期价值。

---

### 3. Bellman 两种方程

#### Bellman Expectation Equation

```math
V^\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma V^\pi(S_{t+1})]
```

含义：

- 评估某个固定策略 `pi` 有多好
- 以后继续按当前 policy 走

#### Bellman Optimality Equation

```math
Q^*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a')]
```

含义：

- 描述最优情况下能有多好
- 下一步默认总选最优动作

速记区分：

- **expectation** = 当前策略有多好
- **optimality** = 理论最优有多好

---

### 4. Bellman Equation vs Bellman Update

#### Bellman Equation

- 描述“正确 value 应该满足什么关系”
- 是理论定义

#### Bellman Update

```math
V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]
```

- 描述“训练时怎么逼近这个关系”
- 是实际更新方法

速记区分：

- **Equation**：定义目标
- **Update**：逼近目标

---

### 5. TD Error

```math
\delta = r + \gamma V(s') - V(s)
```

含义：

- 新证据和旧估计之间的误差
- `delta > 0`：原来估低了
- `delta < 0`：原来估高了

---

### 6. Q-Learning 落地公式

```math
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'}Q(s',a') - Q(s,a)]
```

逐项理解：

- `r`：当前奖励
- `max Q(s',a')`：下一状态最好的未来价值
- 中括号：TD error
- 整体：朝 Bellman target 修正当前 Q 值

---

### 7. 最短背诵版

1. `Reward` 看眼前，`Value` 看长远
2. `Return = 未来奖励的折扣和`
3. `V` 看状态值，`Q` 看动作值
4. `Bellman：长期价值 = 当前收益 + 未来价值`
5. `Expectation` 评估当前策略，`Optimality` 描述最优策略
6. `Q-Learning` 用 Bellman update 反复修正 `Q(s,a)`

---

## 路线图

```
✅ 第一章：RL 是什么（Agent / Environment / State / Action / Reward）
✅ 第二章：Policy π（确定性 vs 随机性）
✅ 第三章：Value Function V(s) 和 Q(s,a)，Bellman 方程
⬜ 第四章：Q-Learning — 自动学出 Q 表的算法
⬜ 第五章：Deep Q-Network (DQN) — 用神经网络替代 Q 表
⬜ 第六章：Policy Gradient — 直接优化策略
⬜ 第七章：Actor-Critic — 结合 Policy 和 Value
```

---

*生成时间：2026-03-11 | 持续更新中 🔥*
