## 第十四章：TD3 - 为什么 DDPG 会被“双 Critic + 延迟更新”继续修正？

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

> 你在这里：连续控制分支 -> TD3

### 序：DDPG 方向对了，但它太容易“自信过头”

DDPG 很重要，但一落地就会暴露出一个经典问题：

> **Critic 往往会把某些动作价值估高，Actor 再去追这些假高分，整个系统就一起飘。**

一句话先压住：

> **TD3 的本质，是承认 DDPG 的不稳定主要来自过估计和耦合过紧，于是用更保守的 Critic 机制和更慢的 Actor 更新来稳住系统。**

---

### 1. Problem：DDPG 为什么容易崩？

#### 1.1 单个 Critic 的误差会被 Actor 放大

如果 Critic 错把某个动作评得很高：
- Actor 会主动朝那个动作靠
- 结果越来越多数据围绕这个假高分区域产生
- 错误被自我强化

#### 1.2 Bellman + function approximation 很容易产生 overestimation

只要：
- 目标值里有 max / 贪心倾向
- 函数逼近有噪声

就容易把价值高估。

在连续控制里，虽然不是显式 `max_a Q(s,a)`，但 Actor 实际上在学“让 Q 尽量大”的动作，所以问题并没有消失。

#### 1.3 Actor 和 Critic 如果同步猛更，会彼此追着跑

Critic 还没评稳，Actor 就跟着动；Actor 一动，Critic 学习分布又变。

所以系统不稳的根源可以压成：

```text
Q 容易高估
+ Actor 会追高估值
+ 两个网络更新耦合太紧
```

---

### 2. Starting Point：如果估值容易虚高，那就主动让 Critic 更保守一点

从第一性原理看，我们真正想要的不是“更乐观的值函数”，而是：

> **宁愿稍微保守一点，也别让 Actor 被假高分带偏。**

于是自然会想到两个修正方向：
- 不要只信一个 Critic
- 不要让 Actor 每一步都追着还没稳定的 Critic 跑

这就是 TD3 的起点。

---

### 3. Invention：TD3 的三个关键补丁是怎么长出来的？

#### 3.1 Twin Critics：两个 Critic 取更小值

TD3 训练两个 Critic：

```text
Q1(s,a), Q2(s,a)
```

构造 target 时取更保守的那个：

```math
y = r + \gamma \min(Q1_{target}(s', a'), Q2_{target}(s', a'))
```

这一步的直觉非常清楚：
- 如果一个 Critic 偶然虚高
- 另一个没那么高
- 取最小值可以压住过估计

#### 3.2 Delayed Policy Update：Actor 更新更慢

TD3 不让 Actor 每一步都更新，而是：
- Critic 多更新几次
- Actor 少更新一次

这相当于在说：

> **先把评分系统校准得更稳一点，再让决策系统跟着动。**

#### 3.3 Target Policy Smoothing：给 target action 也加一点平滑噪声

这是第三个补丁。

在计算目标值时，对 target action 加一点小噪声，可以避免 Critic 把非常尖锐、脆弱的动作峰值当成真最优。

直觉上就是：
- 真正好的策略应该在一个小邻域内都还行
- 不是只对一个极窄动作点突然爆高分

#### 3.4 一句话压缩

```text
TD3 = 双 Critic 压过估计
    + 延迟 Actor 更新降耦合
    + target action 平滑防尖峰骗分
```

---

### 4. Verification：这些补丁为什么真的有效？

#### 4.1 双 Critic 为什么能减轻过估计？

因为高估通常是噪声往上偏造成的。
取两个值里更小的那个，会让 target 更保守。

#### 4.2 延迟更新为什么有用？

因为 Actor 不再每一步都追着不稳定的 Critic 跑。

这样可以降低：
- 错误快速放大
- 两个网络相互追逐导致震荡

#### 4.3 平滑 target action 为什么重要？

因为它在防一种很危险的情况：
- Critic 在某个极窄动作点给出异常高值
- Actor 学着钻这个尖峰漏洞

平滑以后，只有周围也不错的动作区域才更可能被认为真优。

#### 4.4 TD3 是不是终点？

也不是。

它依然主要是：
- deterministic policy
- 依赖外加噪声探索

而探索本身仍然可能不够自然。

这会继续逼出 `SAC`。

---

### 5. Example：自动驾驶方向盘控制

状态：
- 车速
- 横向偏移
- 航向角误差
- 道路线信息

动作：
- 连续方向盘角度

如果 Critic 偶然觉得某个很极端的小转角特别值钱：
- DDPG 的 Actor 可能就会猛追过去
- 导致控制忽左忽右

TD3 用：
- 双 Critic 压虚高
- 延迟更新防追涨
- 平滑 target 防尖峰骗分

于是训练更稳。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
DDPG 的单 Critic 容易高估，Actor 又会追着假高分跑

Starting Point:
宁愿更保守，也别让错误价值被快速放大

Invention:
双 Critic 取较小值 + 延迟 Actor 更新 + target action smoothing

Verification:
这些补丁共同降低过估计和耦合震荡
让连续控制训练更稳定

Example:
方向盘连续控制中，避免追逐错误尖峰动作
```

### 本章速记卡片

#### 一句话主线

- `DDPG` 的核心问题是过估计和耦合过紧
- `TD3` 用三个补丁一起修它
- 目标不是更激进，而是更保守、更稳
- 所以 TD3 常被看作 DDPG 的稳定增强版

#### 必背术语

- `Twin Critics`
- `Delayed Policy Update`
- `Target Policy Smoothing`
- `Overestimation`
- `TD3`

#### 最短背诵版

1. 两个 Critic 比一个更稳
2. Actor 不要每步都更新
3. target action 也要做平滑
4. 核心目的是压过估计
5. TD3 比 DDPG 更稳定

---

