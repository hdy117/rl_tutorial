## 第十四章：TD3 - 为什么 DDPG 会被"双 Critic + 延迟更新"继续修正？

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

### 序：DDPG 的单 Critic + 同步更新会逼出什么矛盾？

DDPG 解决了连续动作空间的问题，但落地时暴露出一个致命缺陷：

> **Critic 高估某些动作的价值 → Actor 追着假高分跑 → 错误被自我强化**

TD3 的核心洞察是：

> **当 DDPG 的过估计和耦合问题积累到一定程度，系统会被迫引入更保守的 Critic 机制和延迟更新来稳住训练。**

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

在连续控制里，虽然不是显式 `max_a Q(s,a)`，但 Actor 实际上在学"让 Q 尽量大"的动作，所以问题并没有消失。

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

从第一性原理看，我们真正想要的不是"更乐观的值函数"，而是：

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

### 5. Example：一个具体的方向盘控制场景

**场景设定：**

```text
状态 s = [车速，横向偏移，航向角误差]
动作 a = 方向盘转角 (-30° ~ +30°)
目标：沿车道中心线行驶
```

**DDPG 会怎么崩？**

假设在某个弯道场景下：
1. Critic 偶然给 `a=25°` 这个极端角度打了一个虚高分数（可能是噪声）
2. Actor 学着往 25° 方向猛转方向盘
3. 结果车辆失控，但系统已经学会了"极端转角=高分"的错误映射

**TD3 怎么稳住？**

1. **双 Critic：** Q1 说 25°值得 80 分，Q2 说只值 40 分 → 取 40 分
2. **延迟更新：** Actor 等 Critic 多跑几轮校准后再动
3. **平滑 target：** 给 25°加噪声后平均打分，发现周围动作都差 → 这个尖峰是假的

**结果：** 训练曲线更稳，最终策略不会追逐虚假高分。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
DDPG 的单 Critic + 同步更新 → 过估计被 Actor 快速放大

Starting Point:
宁愿更保守，也别让错误价值被系统自我强化

Invention:
双 Critic 取较小值 + 延迟 Actor 更新 + target action smoothing

Verification:
训练曲线更稳，收敛率提升，对超参数鲁棒性增强

Example:
方向盘控制中，避免追逐虚假的极端角度尖峰
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

