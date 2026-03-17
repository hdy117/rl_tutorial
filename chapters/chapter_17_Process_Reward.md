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

### 序：如果推理质量决定泛化性，那训练信号就不能只在终点出现

outcome reward 有一个致命的盲区：它无法区分"真的会"和"碰巧对"。

在数学题里，模型可能中间乱推、最后蒙对。
在代码生成里，逻辑极脆但恰好过测试。

如果你只看 outcome，这些投机行为都会拿到高 reward —— 而训练出来的模型会学会投机，而不是推理。

**Process Reward 的必然性从这里开始：**

> 如果最终能力来自过程质量，那训练信号就必须作用在过程上。

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

> **能不能让"中间推理步骤"本身也获得奖励或惩罚？**

---

### 2. Starting Point：从第一性原理看，为什么 outcome reward 必然失效？

**已知事实：**
1. 最终答案由中间推理步骤生成（因果链）
2. Reward 梯度通过时间/序列反向传播到策略更新
3. Credit assignment 的粒度决定了哪些行为会被强化

**推导矛盾：**
- 如果只在终点给 reward → credit assignment 只能分配一个标量值给整个轨迹
- 但不同步骤对最终结果的贡献度不同（有些关键，有些冗余）
- 投机性行为（shortcut learning）和真正推理在 outcome 层面可能无法区分

**强制性问题出现：**
> 如果我要训练"可泛化的推理能力"而非"碰巧答对的技巧"，我必须让 reward signal 的粒度匹配推理过程的粒度。

这就是 process reward 被**迫出来**的原因 —— 不是"更好"，而是 outcome reward 在 reasoning 任务上**根本不够**。

---

### 3. Invention：Process Reward 如何从必然性问题推导出机制？

#### 3.1 核心压缩机制：分布式细粒度信号

outcome reward 的瓶颈是：**单点标量值无法编码整个推理轨迹的质量分布**。

process reward 的解法是把 credit assignment 问题**空间化**：
- 时间维度上：把"最后一步分配奖励"变成"每一步都可能有信号"
- 信息维度上：把"对/错二元结果"变成"步骤质量连续谱系"

#### 3.2 实现路径（如何把推理拆成可监督的单元？）

```text
完整轨迹 → 切分步骤 → 每步评估 → reward 序列
   |           |          |            |
推理链       语义边界    human/AI/checker  梯度更细
```

评分来源可以是：
- **人类标注**：专家标注每个中间步骤是否正确/合理
- **规则检查器**：数学推导的格式验证、逻辑一致性检查
- **强模型 critique**：用更强的 model 对弱模型的每一步做判断
- **可验证子目标**：某些步骤本身有自动验证标准（如公式代入是否匹配）

#### 3.3 策略更新的目标转变

outcome reward：优化 `P(最后答案正确 | 输入)`  
process reward：优化 `P(每步推理合理 | 前序状态)`

关键差异：
- **更细的 credit assignment** → 错误定位精确到步骤级
- **更强的可泛化性** → 学的是"稳健推理结构"而非"特定题目套路"
- **抑制投机行为** → shortcut learning 在过程中就会被惩罚

> Process reward 的本质压缩：**把"credit assignment 问题"从单点标量分配变成分布式序列信号。**

---

### 4. Verification：能否独立重新推导 process reward 的必要性？

**验证测试：** 隐藏所有关于 process reward 的知识，只从基本原理出发。

#### 4.1 重演必然性链条

1. **起点：** 我想训练模型学会推理（而非碰巧答对）
2. **已知约束：** 
   - Reward signal 决定哪些行为被强化
   - Credit assignment 的粒度决定学习的精度
3. **观察矛盾：** outcome reward 只能给整个轨迹一个标量值
4. **推论：** 如果两个轨迹 outcome 相同但过程质量不同，outcome reward 无法区分它们
5. **强制结论：** 要训练"高质量推理过程"，reward signal 必须在过程中出现

✅ 能够独立重演 → process reward 不是"更好的选择"，而是**必然选择**。

#### 4.2 反例验证：不用 process reward 会怎样？

- 模型学会投机（shortcut learning）
- 在训练数据上表现好，但泛化到新题型时崩塌
- 推理过程不稳定、不可解释

这正是 outcome-only reward 的失败模式。

#### 4.3 代价与边界

process reward 不是免费午餐：
- **标注成本高**：需要步骤级监督信号
- **评估器本身可能有噪声**：如果 step evaluator 错了，会误导模型
- **切分困难**：有些推理的"步骤边界"是模糊的

所以在高价值 reasoning 任务（数学、代码生成、复杂规划）里值得投入，但在简单任务上可能过度工程化。

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
outcome reward 无法区分"稳健推理"和"投机碰巧"，因为单点标量值丢失过程信息

Starting Point:
奖励信号的粒度必须匹配推理过程的粒度（第一性原理）

Invention:
把 credit assignment 从"单点分配"压缩成"分布式序列信号"

Verification:
能独立重演必然性链条：要训练可泛化推理 → reward 必须在过程中出现

Example:
数学题里，过程正确的解法 vs 碰巧蒙对答案，process reward 能区分它们
```

### 本章速记卡片

#### 一句话主线

- outcome reward 只能给整个轨迹一个标量值
- process reward 让每一步推理都能获得信号
- 它训练的是"可泛化的推理结构"而非"碰巧答对的技巧"
- 代价：标注成本高、评估器可能有噪声、步骤切分困难

#### 必背术语

- `Process Reward` - 过程级奖励信号
- `Step-Level Signal` - 步骤粒度的监督信号  
- `Credit Assignment Granularity` - credit assignment 的粒度问题
- `Shortcut Learning` - 投机性行为（被 process reward 抑制）
- `Process Supervision` - 过程监督范式

#### 最短背诵版（5 行压缩）

1. outcome reward 无法区分"真会"和"碰巧对"
2. credit assignment 粒度必须匹配推理过程粒度
3. process reward = 分布式细粒度信号替代单点标量值
4. 训练可泛化推理结构，抑制 shortcut learning
5. 代价高但必要性强（不是更好，是 outcome-only 根本不够）

---

