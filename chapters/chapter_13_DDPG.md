## 第十三章：DDPG - 为什么连续动作会逼出“确定性 Actor-Critic”？

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

> 你在这里：连续控制分支 -> DDPG

### 序：PPO 能做连续动作，但为什么工程上还会继续长出 DDPG？

到这里，连续控制分支也该正式接上了。

一个自然问题是：

> 既然 Policy Gradient / Actor-Critic / PPO 都能处理连续动作，为什么还需要 DDPG？

因为连续控制场景还留下了一个非常现实的压力：

> **动作是连续的，采样又贵，如果每次都用随机策略慢慢试，样本效率可能不够。**

一句话先压住：

> **DDPG 的本质，是把 DQN 的 off-policy 数据复用能力，和 Actor-Critic 的连续动作表达能力拼在一起。**

---

### 1. Problem：PPO/随机策略路线在连续控制里哪里不够？

#### 1.1 连续动作空间没法像 DQN 那样直接 `argmax`

这正是前面 Policy Gradient 出现的原因之一。

#### 1.2 但纯随机策略方法样本效率经常不够高

在机器人控制里，数据非常贵：
- 采样慢
- 真实环境代价高
- 仿真也不一定便宜

如果每次都主要依赖 on-policy 新轨迹，成本会很大。

#### 1.3 我们其实很想保留 DQN 那种 replay buffer + off-policy 学习能力

因为 DQN 的一个巨大优点是：
- 老数据也能反复学
- 样本可以高复用

所以连续控制的矛盾变成：

```text
我既要能输出连续动作
又想像 DQN 那样高效复用经验
```

这就自然逼出 DDPG。

---

### 2. Starting Point：能不能让 Actor 直接输出连续动作，而 Critic 继续学 Q？

从第一性原理看，最自然的想法是：

- 既然 `argmax_a Q(s,a)` 在连续空间里难做
- 那不如训练一个 Actor，直接给出“当前最该采取的动作”

于是会出现这样的分工：

#### Actor
直接输出动作：

```math
a = \mu_\theta(s)
```

这里 `\mu_\theta(s)` 是确定性策略。

#### Critic
学习动作价值：

```math
Q_w(s,a)
```

它回答：
- 在状态 `s` 下，采取动作 `a` 值多少钱

于是问题就被改写成：

```text
不用在连续动作空间上做暴力 argmax
而是让 Actor 直接学会近似 argmax 的动作输出
```

---

### 3. Invention：DDPG 是怎么拼出来的？

#### 3.1 把 DQN 的 Bellman 训练方式保留给 Critic

Critic 仍然可以像 Q-learning 那样做 Bellman 回归：

```math
y = r + \gamma Q_{target}(s', \mu_{target}(s'))
```

然后让：

```math
L_{critic} = (Q(s,a) - y)^2
```

这一步说明 DDPG 没有丢掉 value-based 的核心资产：
- bootstrapping
- target network
- replay buffer
- off-policy 学习

#### 3.2 让 Actor 直接朝着让 Q 变大的方向更新

既然 Critic 已经会评估动作值，那 Actor 最自然的目标就是：

> **输出一个能让 Critic 评分更高的动作。**

所以 Actor 会按 `Q(s, \mu(s))` 的梯度方向更新。

直觉上就是：
- Critic 说这个动作值更高
- 那 Actor 就把输出往这个方向推

#### 3.3 为什么叫 deterministic？

因为它不是输出动作分布再采样，而是直接输出一个确定动作：

```math
a = \mu_\theta(s)
```

探索通常靠额外加噪声完成，比如：
- Gaussian noise
- OU noise

也就是说：
- 训练时：动作 = Actor 输出 + 探索噪声
- 部署时：动作 = Actor 直接输出

#### 3.4 一句话压缩

```text
DDPG = DQN 的 off-policy / replay / target 思想
     + Actor-Critic 的连续动作表达
     + 确定性策略输出
```

---

### 4. Verification：DDPG 真的解决了连续控制里的关键矛盾吗？

#### 4.1 它能不能处理连续动作？

能。

因为 Actor 直接输出连续值，不再需要离散化动作。

#### 4.2 它有没有保留高样本效率？

有一部分。

因为它是 off-policy：
- 旧经验可以重复利用
- replay buffer 很重要

这通常比纯 on-policy 方法更省样本。

#### 4.3 它有没有新的问题？

有，而且不小：
- Critic 容易过估计
- Actor 可能被错误 Critic 带偏
- 训练对超参数很敏感
- 探索质量也常常不稳定

所以 DDPG 是一个很重要的桥，但不是终点。

这就自然过渡到下一章：`TD3`。

---

### 5. Example：机械臂控制连续扭矩

状态：
- 机械臂关节角
- 角速度
- 末端与目标位置关系

动作：
- 每个关节的连续扭矩

如果动作离散化：
- 很粗糙
- 组合爆炸

DDPG 的做法是：
- Actor 直接输出一组连续扭矩
- Critic 评价这组扭矩值不值钱
- replay buffer 反复复用过往经验

这就很适合“动作连续且样本贵”的任务。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
连续动作没法直接 argmax，但 on-policy 随机策略又可能太费样本

Starting Point:
让 Actor 直接输出动作，让 Critic 继续学 Q

Invention:
用 deterministic policy 输出连续动作
保留 replay buffer / target network / off-policy Q-learning 思想

Verification:
它兼顾了连续动作表达和较高样本复用
但稳定性仍然不够，容易过估计

Example:
机械臂连续扭矩控制
```

### 本章速记卡片

#### 一句话主线

- 连续动作逼出了 Actor
- 样本效率需求又逼回了 DQN 的 off-policy 思想
- `DDPG` 就是这两条线的拼接
- 它能做连续控制，但稳定性还有问题

#### 必背术语

- `Deterministic Policy`
- `Replay Buffer`
- `Target Network`
- `Off-Policy`
- `DDPG`

#### 最短背诵版

1. Actor 直接输出连续动作
2. Critic 评估这个动作值多少钱
3. 老经验可以反复学
4. 所以样本效率较高
5. 但训练容易不稳定

---

