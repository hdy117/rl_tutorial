## 第十八章：Verifiable Reward - 为什么 reasoning 训练越来越依赖"可自动检查"的中间与最终信号？

### 📍 定位

> Outcome Reward → Process Reward → **Verifiable Reward**

本章聚焦：如何把 reward 变成可验证、可自动化、可规模扩展的信号。

### 序：最理想的 reward，不只是强，还要便宜、稳定、可扩展

LLM reasoning post-training 正在朝一个方向收敛：

> **尽量把 reward 变成可验证、可自动化、可规模扩展的信号。**

人类偏好、outcome reward、process reward——都在追求同一个目标：**给模型稳定、可信、可大规模供应的训练反馈**。

核心原则：

> **Verifiable Reward = 用可自动检查的事实约束代替主观印象**

---

### 1. Problem：为什么 reasoning 训练越来越偏向 verifiable reward？

**人工标注的瓶颈：**
- 贵、慢、一致性差

**复杂推理任务的需求：**
数学、编程、定理证明、工具使用——这些场景需要海量训练数据，人工覆盖不过来。

**结论：** reward 必须可自动判、可重复判、规则明确。这样才能：
- 更稳
- 更便宜  
- 更可扩展

于是"verifiable"成了 reward 设计的核心标准。

---

### 2. Starting Point：任务能形式化，就把判分外包给 verifier

奖励的职责不是"优雅地描述任务"，而是：**稳定地区分更好和更差的行为**。

verifier 可以作用在：
- 最终答案
- 中间步骤  
- 格式约束
- 工具调用结果
- 外部环境反馈

所以：

```text
Verifiable reward = 一种 reward 设计原则，不是单一算法
```

---

### 3. Invention：Verifiable Reward 在实践里怎么落地？

#### 3.1 最终结果检查
数学答案比对、单元测试、执行结果比对。

#### 3.2 中间过程检查  
子步骤公式是否成立、中间程序状态是否正确、推理链条是否满足局部规则。

#### 3.3 结构与格式检查
JSON 是否合法、工具调用参数是否匹配 schema、是否遵守输出协议。

#### 3.4 混合奖励（最常见）
实践里很少单一 reward，通常是：
- 偏好 reward + outcome reward + process reward + verifier signal

这层底座原则很简单：**凡是能自动判的地方，就尽量别只靠主观打分。**

---

### 4. Verification：这条路为什么重要？有边界吗？

**好处：**
- 信号更稳（规则检查比人类印象一致）
- 扩展性更强（verifier 能批量跑，数据规模大很多）

**边界：**
不是所有任务都容易验证。开放式写作、审美表达、长篇创意——这些仍需偏好或人工判断。

结论：在能验证的任务上，verifiable reward 是主力；在难验证的任务上，它和偏好信号并存。

---

### 5. Example：现代 reasoning 训练管线的真实形态

代码智能体训练例子：
- **Layer 1**：输出格式是否合法（JSON/schema）
- **Layer 2**：跑单测看最终行为是否正确
- **Layer 3**：检查关键中间步骤是否满足约束  
- **Layer 4**（可选）：偏好模型评估可读性和帮助性

reward 不再是单一分数，而是 layered system：格式正确性 + 过程正确性 + 结果正确性 + 人类偏好。这就是 verifiable reward 思路的现代形态。

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

