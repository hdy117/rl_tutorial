## 第十八章：Verifiable Reward - 为什么 reasoning 训练越来越依赖“可自动检查”的中间与最终信号？

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

> 你在这里：LLM reasoning reward 分支 -> Verifiable Reward

### 序：最理想的 reward，不只是强，还要便宜、稳定、可扩展

到了这里，你会发现 LLM reasoning post-training 正在朝一个非常自然的方向收敛：

> **尽量把 reward 变成可验证、可自动化、可规模扩展的信号。**

因为无论是：
- 人类偏好
- outcome reward
- process reward

最终都在追求一个共同目标：

> 给模型稳定、可信、可大规模供应的训练反馈

一句话先压住：

> **Verifiable Reward 的本质，是尽量把奖励建立在可自动检查的事实约束上，而不是纯主观印象上。**

---

### 1. Problem：为什么 reasoning 训练会越来越偏向 verifiable reward？

#### 1.1 人类标注贵且不稳定

纯靠人工偏好：
- 成本高
- 吞吐低
- 一致性有限

#### 1.2 复杂推理任务需要高密度、高可信信号

如果任务是：
- 数学
- 编程
- 定理证明
- 工具使用

那训练量非常大，人工很难覆盖。

#### 1.3 所以 reward 最理想的形态是：可自动判、可重复判、规则明确

这会让训练变得：
- 更稳
- 更便宜
- 更可扩展

于是“verifiable”本身变成了 reward 设计里的一个核心标准。

---

### 2. Starting Point：只要任务能部分形式化，就该尽量把正确性外包给 verifier

从第一性原理看，奖励的职责不是“优雅地描述任务”，而是：

> **稳定地区分更好和更差的行为。**

如果 verifier 能做到这一点，那就应该尽量使用 verifier。

verifier 可以作用在：
- 最终答案
- 中间步骤
- 格式约束
- 工具调用结果
- 外部环境反馈

也就是说：

```text
Verifiable reward 不是单一算法
而是一种 reward 设计原则
```

---

### 3. Invention：Verifiable Reward 在实践里通常怎么出现？

#### 3.1 最终结果检查

比如：
- 数学答案比对
- 单元测试
- 执行结果比对

#### 3.2 中间过程检查

比如：
- 子步骤公式是否成立
- 中间程序状态是否正确
- 推理链条是否满足局部规则

#### 3.3 结构与格式检查

比如：
- JSON 是否合法
- 工具调用参数是否匹配 schema
- 是否遵守输出协议

#### 3.4 混合奖励

实践里常常不是单一 reward，而是：
- 偏好 reward
- outcome reward
- process reward
- verifier signal

一起组合。

这说明 verifiable reward 更像一层底座：

> **凡是能自动判的地方，就尽量别只靠主观打分。**

---

### 4. Verification：为什么这条路对未来 reasoning 训练特别重要？

#### 4.1 它有没有提高信号稳定性？

有。

因为规则明确的检查通常比人类即时印象更一致。

#### 4.2 它有没有提高扩展性？

有。

只要 verifier 能批量运行，数据规模就能大很多。

#### 4.3 它有没有边界？

当然有。

不是所有任务都容易验证：
- 开放式写作
- 审美表达
- 长篇创意任务

这些仍然需要偏好或人工判断。

所以最准确的结论不是“verifiable reward 将替代一切”，而是：

> **在能验证的任务上，verifiable reward 会越来越成为主力；在难验证的任务上，它会和偏好信号并存。**

---

### 5. Example：一个真正现代的 reasoning 训练管线长什么样？

例如代码智能体训练：
- 先看输出格式是否合法
- 再跑单测看最终行为是否正确
- 再检查关键中间步骤是否满足约束
- 必要时叠加偏好模型去评估可读性和帮助性

这时 reward 就不是单一分数，而是一个 layered system：
- 格式正确性
- 过程正确性
- 最终结果正确性
- 人类偏好

这就是 verifiable reward 思路的现代形态。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
reasoning 训练需要大规模、稳定、可信的奖励信号，纯人工偏好不够便宜也不够稳

Starting Point:
只要任务能部分形式化，就应该尽量把判分交给 verifier

Invention:
把最终结果检查、过程检查、格式检查等自动信号都纳入 reward 设计

Verification:
这提高了稳定性和扩展性
但开放式任务仍需要偏好或人工信号补充

Example:
代码智能体训练同时结合格式、过程、结果与偏好四层奖励
```

### 本章速记卡片

#### 一句话主线

- `Verifiable Reward` 不是单一算法，而是一种 reward 设计原则
- 能自动检查的地方，就尽量交给 verifier
- 它让 reasoning 训练更稳、更便宜、更可扩展
- 但开放式任务仍然需要偏好信号配合

#### 必背术语

- `Verifier`
- `Verifiable Reward`
- `Outcome Signal`
- `Process Signal`
- `Hybrid Reward`

#### 最短背诵版

1. 奖励最好能自动检查
2. 最终结果、过程、格式都可以验
3. 这样训练更稳定、更可扩展
4. 但不是所有任务都能完全验证
5. 所以 verifier 和偏好通常会并存

---

