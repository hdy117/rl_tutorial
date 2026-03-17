## 第十一章：PPO for LLMs - 当动作变成 token，PPO 还在优化什么？

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

> 你在这里：LLM 分支 -> PPO for LLMs

### 序：PPO 进了语言模型，不是换个名字，而是换了“状态、动作、奖励”的语义

到了 LLM 分支，很多人会有一个错觉：

> PPO for LLMs 不就是把 PPO 原封不动搬过来吗？

不对。核心思想没变，但语义全变了。

在经典控制里：
- 状态可能是机器人姿态
- 动作可能是关节扭矩
- 奖励可能来自环境反馈

在 LLM 里：
- 状态是 `prompt + 已生成前缀`
- 动作是“下一个 token 选什么”
- 一整段回答生成完以后，才更容易拿到回答级 reward

先把这一章的主线钉住：

```text
problem -> starting point -> invention -> verification -> example
```

一句话先压住：

> **PPO for LLMs 的本质，是把语言模型当作 token-level policy，在 reward model 的指导下做保守更新，同时用 KL 约束防止模型偏离原本可用语言分布太远。**

---

### 1. Problem：为什么 SFT 之后还会继续逼出 PPO for LLMs？

`SFT` 已经让模型进入了“像助手一样回答”的接口，但它仍然有三个明显边界。

#### 1.1 SFT 学的是 imitation，不是 preference optimization

在 `SFT` 里，模型主要学的是：
- 给定 prompt
- 模仿高质量 answer 的 token 序列

这当然有用，但它本质还是：
- **模仿已有示范**

而不是：
- **主动优化回答级偏好目标**

#### 1.2 很多偏好不是唯一标准答案

比如：
- 更简洁
- 更稳妥
- 更 helpful
- 更不容易胡说

这些往往不是“只有一个 gold answer”，而是：
- 多个回答都可以
- 但质量高低不同

这种信号更像 reward，而不像标准 supervised label。

#### 1.3 如果直接只追 reward，又会把模型带偏

一旦有了 reward model，你会自然想：
- 那就让模型猛追高 reward 不就行了？

问题是 reward model 本身并不完美：
- 可能 noisy
- 可能有 shortcut
- 可能被 exploit

如果策略更新太猛，模型就会：
- 套路化迎合 reward model
- 语言分布变形
- 丢掉原本预训练 + SFT 积累的通用能力

所以问题可以压成一句：

> **我们既想让 LLM 朝高偏好回答移动，又不能让它为了刷 reward 而一步走偏。**

---

### 2. Starting Point：把 LLM 重新翻译成 RL 语言

想让 PPO 接上来，第一步不是写公式，而是完成语义映射。

#### 2.1 状态是什么？

在第 `t` 个生成位置，状态可以理解成：

```text
s_t = prompt + 已生成 token 前缀
```

也就是说，当前上下文本身就是 state。

#### 2.2 动作是什么？

动作是：

```text
a_t = 选择下一个 token
```

这和机器人控制最大的不同在于：
- 动作空间极大（整个词表）
- 一次回答是很多步 token action 串起来的轨迹

#### 2.3 策略是什么？

策略就是语言模型的条件分布：

```math
\pi_\theta(a_t|s_t) = p_\theta(token_t | prompt, token_{<t})
```

这一步一旦看清，前面的 RL 主线就能重新接上：

```text
LLM 不是例外
它只是一个动作空间极大、轨迹很长、奖励更偏回答级的策略模型
```

#### 2.4 那奖励从哪来？

在 LLM 场景里，奖励通常不是每个 token 都有环境反馈，而更像：
- 整个回答生成完以后
- 由 reward model 给分
- 再可能结合规则奖励 / 安全惩罚 / 格式奖励

所以它更像序列级 reward，再反向分配到 token 轨迹上。

---

### 3. Invention：PPO for LLMs 到底是怎么工作的？

#### 3.1 先采样回答，再让 reward model 评分

典型流程是：
1. 取一批 prompts
2. 用当前 policy 采样 responses
3. 用 reward model 给每条 response 打分
4. 构造 advantage / return 信号
5. 用 PPO 更新 policy

也就是说，PPO 在这里做的不是“在线操纵环境”，而是：

> **对一批已生成回答做 post-hoc 评价，再反过来更新生成策略。**

#### 3.2 为什么还要 reference model / KL 约束？

这是 LLM 版 PPO 的关键补丁。

因为仅靠 reward 最大化，模型很容易走向：
- 怪异措辞
- 套模板刷分
- 极端自信
- 语言分布脱离原始模型

所以实践里通常会保留一个 `reference model`，并加入 KL penalty，大意是：

```text
你可以朝高 reward 方向改
但不要离原始 SFT policy 太远
```

这相当于在 LLM 里又多了一层“语言分布护栏”。

#### 3.3 为什么在 LLM 里，PPO 看起来不像经典 control 里的 PPO？

因为它处理的是：
- 长序列 token 轨迹
- 序列级奖励
- 超大动作空间
- 参考分布约束

所以公式精神没变，但工程形态已经明显变了：
- Actor 还是 policy
- 可能还有 value head / critic
- 但训练流程高度依赖 batch responses、reward model、KL regularization

#### 3.4 一句话压缩

```text
PPO for LLMs = 回答采样 + 偏好打分 + 保守策略更新 + KL 防跑偏
```

---

### 4. Verification：为什么 PPO for LLMs 比“只做 SFT”或“只追 reward”更合理？

#### 4.1 它有没有把回答级偏好真正接进优化目标？

有。

相比 SFT 只模仿示范，PPO for LLMs 让模型直接面对：
- 哪类完整回答 reward 更高

所以偏好不再只是数据集里的隐含风格，而是显式优化信号。

#### 4.2 它有没有避免模型因为 reward noisy 而剧烈跑偏？

有，至少方向上比裸 policy gradient 更稳。

因为它同时用了：
- PPO 的保守更新
- KL 对 reference policy 的约束

这两层都在防止：
- 一次高分样本把模型带疯

#### 4.3 它是不是最终答案？

也不是。

它仍然有很多成本：
- rollout 很贵
- reward model 训练很贵
- value / critic 也可能难训
- KL 系数和 reward scale 很敏感

这也是为什么后面会继续逼出更贴近 LLM 场景的变体，比如 `GRPO`。

---

### 5. Example：同一个 prompt 下，PPO for LLMs 在优化什么？

假设 prompt 是：

```text
请解释 Bellman 方程，并给一个最小例子。
```

模型采样出两条回答：

- 回答 A：结构清楚，先讲定义，再讲递推，再给小例子
- 回答 B：术语很多，但解释混乱，例子也不落地

reward model 可能给出：
- `R(A) > R(B)`

PPO for LLMs 不会只说“把 A 的每个 token 当标准答案死记住”，而是更像：

```text
这类完整回答整体更符合偏好
-> 让生成这类回答的 token 决策轨迹整体更容易再次出现
-> 但不要一步把模型整体语言分布改坏
```

这就是它和纯 SFT 的本质区别。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
SFT 只能模仿示范，reward model 又可能 noisy，直接追 reward 容易把模型带偏

Starting Point:
把 LLM 重新翻译成 RL：上下文是 state，下一个 token 是 action

Invention:
采样完整回答 -> reward model 打分 -> PPO 保守更新
并用 reference model + KL 限制偏离原始分布过远

Verification:
它比只做 imitation 更能对齐回答级偏好
也比裸追 reward 更稳

Example:
对于同一个 prompt，模型会逐渐更偏向高质量回答的整条 token 决策轨迹
```

### 本章速记卡片

#### 一句话主线

- `LLM` 也可以看成 policy
- `token` 是 action，完整回答是一条轨迹
- `reward model` 给回答级偏好信号
- `PPO + KL` 负责保守地推动策略朝高偏好回答移动

#### 必背术语

- `Token Policy`
- `Reward Model`
- `Reference Model`
- `KL Penalty`
- `PPO for LLMs`

#### 最短背诵版

1. LLM 也是策略
2. 每个 token 都是动作
3. reward 来自完整回答质量
4. PPO 负责保守更新
5. KL 负责别跑偏

---

