## 第十七章：Process Reward - 为什么只奖最终答案还不够？

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

> 你在这里：LLM reasoning reward 分支 -> Process Reward

### 序：如果过程本身决定了可泛化性，那训练就不能只在终点打分

一旦你开始用 outcome reward，很快就会撞上一个更细的问题：

> **两个答案都对，不代表它们的推理质量一样。**

有的回答：
- 过程清晰
- 步步正确
- 可泛化

有的回答：
- 只是碰巧蒙对
- 中间漏洞很多
- 不稳定，不可迁移

一句话先压住：

> **Process Reward 的本质，是把奖励从“只看最后结果”推进到“中间推理步骤本身也要被评价”。**

---

### 1. Problem：Outcome Reward 为什么还不够？

#### 1.1 最终结果对，不等于推理过程好

在数学题里，模型可能：
- 中间乱推
- 最后碰巧写对

在代码里，模型可能：
- 逻辑极脆
- 恰好过了当前测试

如果你只看 outcome，那这些都可能拿到高 reward。

#### 1.2 只看终点，credit assignment 太粗

如果最后错了，你不知道：
- 是第 2 步错了
- 还是第 8 步错了
- 还是前面都对，最后抄错了数字

所以 outcome reward 虽然硬，但信息太晚、太粗。

#### 1.3 reasoning 能力的关键，常常在中间过程

如果我们真正想让模型学会：
- 拆问题
- 保持逻辑一致
- 逐步验证
- 少走歪路

那训练信号就不能只在最后一秒钟出现。

于是问题变成：

> **能不能让“中间推理步骤”本身也获得奖励或惩罚？**

---

### 2. Starting Point：如果最终成功来自一串中间步骤，那好步骤也应该被显式强化

从第一性原理看，最终结果是由过程生成的。

所以如果我们只奖终点，不奖过程，就等于：
- 只知道谁赢了
- 但不知道赢法值不值得复用

更自然的目标应该是：

```text
最终结果要对
中间步骤也要合理
```

这意味着 reward 结构要细化成：
- 结果级 reward
- 过程级 reward

---

### 3. Invention：Process Reward 是怎么工作的？

#### 3.1 把一条推理轨迹拆成若干步骤

例如数学解题过程可以拆成：
- 列公式
- 代入
- 化简
- 求根
- 检查答案

#### 3.2 对中间步骤打分

评分方式可能来自：
- 人类标注步骤好坏
- 规则检查器
- 更强模型做 step critique
- 自动一致性 / 可验证子目标检查

于是 reward 不再只有最后一个终点值，而是可以在中间多次出现。

#### 3.3 让策略不只学“最后答对”，还学“怎么一步步更稳地答对”

这一步的真正收益在于：
- 更细的 credit assignment
- 更强的可泛化性
- 更少的投机性 shortcut

也就是说，process reward 在优化的是：

> **可复用的推理过程结构。**

---

### 4. Verification：为什么 process reward 对 reasoning 特别关键？

#### 4.1 它有没有改善 credit assignment？

有。

因为错误和正确现在都可以更早、更局部地被指出。

#### 4.2 它有没有帮助模型学到更稳的推理结构？

通常有。

因为它不再只奖励“最后碰巧对”，而是更偏向：
- 中间也合理
- 步骤也一致
- 每一步都更可解释

#### 4.3 它有没有代价？

当然有：
- 过程标注贵
- 步骤切分难
- 过程评估本身也可能 noisy
- 错误的过程监督可能反而误导模型

所以它不是白送的午餐，但在高价值 reasoning 任务里，往往非常值得。

---

### 5. Example：数学题里的 outcome vs process

题目：解方程 `x^2 - 5x + 6 = 0`

模型 A：
- 中间因式分解正确
- 推理清晰
- 最后答案正确

模型 B：
- 中间乱写
- 最后碰巧给出 `x=2,3`

如果只看 outcome：
- A 和 B 都可能得高分

如果加上 process reward：
- A 会明显更高
- B 会被压下去

这就是为什么 process reward 更适合训练真正稳的 reasoning。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
outcome reward 只看终点，无法充分区分“稳健推理”和“碰巧做对”

Starting Point:
如果结果来自过程，那过程中的好步骤也应该被强化

Invention:
把推理拆成步骤，对中间步骤打 reward / critique / verifier signal

Verification:
它改善了 credit assignment，也更利于学到可泛化的 reasoning 结构

Example:
数学题里，过程正确的解法应当比“蒙对答案”的解法得更高奖励
```

### 本章速记卡片

#### 一句话主线

- `Outcome Reward` 只看最后结果
- `Process Reward` 还会看中间步骤好不好
- 它更适合训练真正稳定的推理能力
- 但代价是过程标注和评估更难

#### 必背术语

- `Process Reward`
- `Step-Level Signal`
- `Credit Assignment`
- `Reasoning Trace`
- `Process Supervision`

#### 最短背诵版

1. 只看结果不够
2. 过程也要被奖励
3. 这样 credit assignment 更细
4. 更能训练稳定推理
5. 但标注和评估成本更高

---

