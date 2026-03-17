## 第十五章：SAC - 为什么最大熵原则会逼出"既学回报，也保留随机性"的算法？

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

> 你在这里：连续控制分支 -> SAC

### 序：如果探索不该只是"外加噪声"，那随机性就该进目标函数本身

到了 TD3，你会发现它虽然更稳了，但还有一个深层限制：

> **探索主要还是靠外加噪声，而不是策略本身天然愿意保持合理随机性。**

这会逼出一个更根本的问题：

> **能不能把"探索"从训练技巧，升级成优化目标的一部分？**

一句话先压住：

> **SAC 的本质，是不只最大化 reward，还同时鼓励策略保持较高 entropy，让 agent 在学高回报行为时仍然保留有价值的随机性。**

---

### 1. Problem：为什么 DDPG / TD3 之后还会继续逼出 SAC？

#### 1.1 deterministic policy 的探索太依赖手工噪声

在 DDPG / TD3 里，常见做法是：
- Actor 给出一个确定动作
- 再手动加噪声做探索

这当然能工作，但它有个问题：
- 探索和目标函数本身是分离的

也就是说，模型本身并没有学到：
- 什么时候应该更确定
- 什么时候应该保留随机性

#### 1.2 真实任务里，随机性有时不是缺陷，而是资产

有些环境需要：
- 多模态动作选择
- 避免过早陷入局部最优
- 在不确定状态下保留尝试空间

如果策略太早塌缩成单点动作，可能会：
- 探索不足
- 容易卡住

#### 1.3 所以"探索"不该只是外挂

这就把问题逼得很明确：

> **如果随机性本身有价值，那它就应该直接出现在优化目标里。**

---

### 2. Starting Point：既然随机性有价值，就把 entropy 也当成被奖励的对象

从第一性原理看，reward 最大化目标只表达了一件事：
- 追求高回报

但它没有表达：
- 保持策略分布的丰富性

于是最自然的扩展就是：

```text
不仅要高 reward
还要高 entropy
```

熵 `entropy` 在这里可以粗略理解成：
- 策略分布有多分散
- 行为有多不那么死板

所以 SAC 的出发点就是：

> **把"探索价值"写进目标函数，而不是只靠训练时手工加噪声。**

---

### 3. Invention：从 axioms 出发，SAC 为什么必须这样存在？

#### 3.0 第一步：找到不可约的 axioms（基础事实）

```text
Axiom 1: reward maximization → 追求高回报是目标
Axiom 2: exploration is necessary → 不探索就无法发现更好的策略
Axiom 3: noise ≠ intrinsic randomness → 外挂噪声不是策略本身的随机性
```

#### 3.1 第二步：如果只有 axioms，会引出什么矛盾？

**从 Axiom 1 + Axiom 2 出发：**
- reward maximization 会自然导向确定性策略（选最大 Q 值的动作）
- exploration necessary 需要保持随机性
- **矛盾！** 最大化 reward 和保持探索性是内在冲突的

DDPG / TD3 的做法：
```text
确定性 policy → 外挂噪声 → 强行探索
```
但这不是从目标函数本身解决，而是训练技巧层面的补丁。

#### 3.2 第三步：唯一合理的解决方案路径是什么？

如果"探索有价值"这个事实成立（Axiom 2），那有且只有一种方式真正内生化它：

```text
Option A: 外挂噪声 → ❌ 不是目标函数的一部分
Option B: entropy 进目标函数 → ✅ 这才是内生化的唯一路径
```

为什么必须是 Option B？因为：
- 优化器只能优化"被明确表达的目标"
- 如果探索有价值，它就必须出现在 objective 里
- **所以 `maximize reward + entropy` 是唯一合理的推导结果**

#### 3.3 第四步：如何让它可扩展（compression mechanisms）？

纯加 entropy 项会导致数值不稳定。SAC 引入了：

```text
L_SAC = E[reward] + α * H(π)
         ↑           ↑
    reward 目标    entropy 正则化
    
α: temperature parameter → 控制随机性 vs 回报的权衡
```

这就是 SAC 最终形式的推导链条。

#### 3.4 policy 重新变成 stochastic

和 DDPG / TD3 的确定性策略不同：

```text
DDPG/TD3: π(s) → a (确定动作)
SAC:      π(a|s) ~ N(μ, σ²) (概率分布，从中采样)
```

这意味着：
- 探索是 policy 自身的属性
- entropy 可以被显式计算和优化
- 模型学到"什么时候该随机、什么时候该确定"

#### 3.5 Critic 的 Bellman target 也要调整

因为 objective 变了，Bellman 方程也要对应调整：

```text
传统 Q-learning: Q(s,a) = E[r + γQ(s',a')]

SAC Q-learning: Q(s,a) = E[r + γ(Q(s',a') - α log π(a'|s'))]
                               ↑__________________↑
                                 entropy 修正项
```

这说明 SAC 不是"局部修补"，而是从目标函数到 Bellman target 的完整一致性重构。

#### 3.6 一句话压缩推导链

```text
Axiom: reward + exploration are both necessary  
Contradiction: deterministic policy kills exploration  
Solution: entropy must enter objective (only one way)  
Result: SAC = off-policy Actor-Critic with stochastic π + αH(π)
```

#### 3.7 检验理解：能否独立重推？

**试着隐藏上面的推导，自己重建：**

1. DDPG/TD3 的确定性策略有什么根本问题？（探索是外挂）
2. 如果探索有价值，它必须出现在哪里？（目标函数里）
3. 如何表达"随机性价值"？（entropy）
4. 所以 objective 是什么？（reward + α*entropy）
5. policy 要改成什么形式？（stochastic）

**如果能独立推出这 5 步，说明理解了 SAC 的必然性。**

---

### 4. Verification：为什么最大熵这条路确实解决了前面的深层问题？

#### 4.1 它有没有让探索变成目标内生的一部分？

有。

这正是 SAC 和 DDPG / TD3 的最大哲学差别：
- 前者：探索是目标的一部分
- 后者：探索更多是外挂噪声

#### 4.2 它能不能缓解过早收缩到差策略？

通常能。

因为高熵目标会惩罚过快塌缩，让策略在 early stage 保留更多尝试空间。

#### 4.3 它是不是就一定比 TD3 好？

也不是绝对。

不同任务里：
- 有时 TD3 更简单直接
- 有时 SAC 更稳、更强

但从思想层面看，SAC 解决的是一个更深的问题：

> **如何让探索不再只是训练技巧，而成为优化目标的一部分。**

---

### 5. Example：移动机器人导航

状态：
- 位置
- 速度
- 雷达 / 传感器信息
- 目标相对方向

动作：
- 连续线速度
- 连续角速度

如果策略太早确定成一种走法：
- 可能会陷入局部路线
- 遇到新障碍适应差

SAC 会倾向于：
- 在学习高 reward 路径的同时
- 维持一定策略随机性
- 不那么快塌成唯一动作模板

这对复杂环境探索很有价值。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
DDPG / TD3 的探索主要靠外加噪声，随机性没有进入目标函数

Starting Point:
如果随机性本身有价值，那 entropy 就应该被显式奖励

Invention:
最大化 reward + entropy
使用 stochastic policy，并保留 off-policy Actor-Critic 结构

Verification:
探索从外挂技巧变成目标内生部分
能缓解策略过早塌缩

Example:
移动机器人导航时，保持合理随机性有助于找到更稳健路径
```

### 本章速记卡片

#### 一句话主线

- `SAC` 不是只学高 reward
- 它还显式鼓励高 entropy
- 所以探索不再只是外加噪声
- 这让它在很多连续控制任务里又稳又强

#### 必背术语

- `Entropy`
- `Temperature alpha`
- `Stochastic Policy`
- `Maximum Entropy RL`
- `SAC`

#### 最短背诵版

1. 不只要高 reward
2. 还要保留随机性
3. 所以 entropy 直接进目标函数
4. policy 重新变成 stochastic
5. 这就是 SAC 的核心

---

