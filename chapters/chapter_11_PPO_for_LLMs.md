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

### 序：PPO 进了语言模型，不是换个名字，而是换了"状态、动作、奖励"的语义

到了 LLM 分支，很多人会有一个错觉：

> PPO for LLMs 不就是把 PPO 原封不动搬过来吗？

不对。核心思想没变，但语义全变了。

在经典控制里：
- 状态可能是机器人姿态
- 动作可能是关节扭矩
- 奖励可能来自环境反馈

在 LLM 里：
- 状态是 `prompt + 已生成前缀`
- 动作是"下一个 token 选什么"
- 一整段回答生成完以后，才更容易拿到回答级 reward

先把这一章的主线钉住：

```text
problem -> starting point -> invention -> verification -> example
```

一句话先压住：

> **PPO for LLMs 的本质，是把语言模型当作 token-level policy，在 reward model 的指导下做保守更新，同时用 KL 约束防止模型偏离原本可用语言分布太远。**

---

### 1. Problem：为什么 SFT 之后还会继续逼出 PPO for LLMs？

`SFT` 已经让模型进入了"像助手一样回答"的接口，但它仍然有三个明显边界。

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

这些往往不是"只有一个 gold answer"，而是：
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

### 3. Invention：为什么 PPO 的"保守更新+KL 约束"是这个问题的唯一合理解？

#### 3.1 Axioms 层次：这个问题的不可绕过性是什么？

先把最底层的 axioms 钉死：

**Axiom 1 (SFT 的本质)：** SFT 只能优化 token-level likelihood，无法直接优化回答级偏好
**Axiom 2 (Reward Model 的不完美)：** RM 有 noise、可能被 exploit、可能和真实人类偏好有 gap
**Axiom 3 (语言分布的脆弱性)：** LLM 的语言能力是预训练 + SFT 累积的结果，剧烈更新会破坏通用能力

如果只接受这三个 axioms，那么问题就变成：

> **如何在 RM 指导下优化偏好，同时不破坏模型原有的语言能力？**

这不是"要不要加 KL"的问题，而是**KL 约束是必然的解**——因为如果允许策略自由跑向高 reward，最优策略就是"找到 RM 的捷径并疯狂刷分"。

#### 3.2 Starting Point：我们手上有什么工具？

- `PPO` 本身有 clipped surrogate objective —— 这已经是一种保守更新
- `Reference model` π_ref 可以冻结在 SFT checkpoint，作为"语言能力锚点"
- `KL penalty` 可以作为距离度量：D_KL(π_θ || π_ref)

#### 3.3 Invention：为什么 PPO + KL 是 inevitable outcome？

**Step 1: 为什么要用 PPO 而不是 PG？**
- PG 一步更新太猛，容易 overshoot
- PPO 的 clip 机制天然限制单步更新幅度 → **这是保守性的第一层**

**Step 2: 为什么还需要 KL penalty？**
- PPO 只限制"当前 update step 不要太大"
- 但不限制"最终策略可以离 π_ref 多远"
- KL penalty 是**对整体分布距离的约束**——这是第二层护栏

**压缩机制的本质：**
```text
KL(π_θ || π_ref) = E_{x~π_θ}[log π_θ(x) - log π_ref(x)]

如果 KL 很小 → π_θ 和 π_ref 在大部分 token 上决策一致
→ 模型不会为了刷 reward 而发明怪异语言模式
```

**Step 3: 为什么是"参考分布"而不是别的东西？**
- SFT checkpoint 代表了"人类示范 + 通用语言能力"的平衡点
- RM 只代表偏好信号，可能 noisy / biased
- KL constraint 的本质是：**让 RM 指导优化方向，但不让它完全接管策略**

#### 3.4 Verification：能否独立重推导这个设计？

试着遮住答案，自己推一遍：

1. **问题是什么？** — SFT 只能 imitation，RM 能指导偏好但可能 noisy
2. **如果只用 RM + PG 会发生什么？** — 模型会 exploit RM，语言分布崩塌
3. **需要什么机制来防跑偏？** — 保守更新 + 距离约束
4. **PPO 的 clip 够不够？** — 只限制单步，不限制整体分布
5. **还需要什么？** — KL penalty 对参考分布的距离度量
6. **参考分布选哪里？** — SFT checkpoint，因为它有语言能力的积累

**如果你能自己推到"KL constraint is necessary"这个结论，那才真正理解了 PPO for LLMs。**

#### 3.5 Example：同一个 prompt 下发生了什么？

假设 prompt 是：
```text
请解释 Bellman 方程，并给一个最小例子。
```

**SFT 阶段：**
- 模型学的是"模仿高质量回答的 token 序列"
- 优化目标：maximize likelihood of gold answers

**PPO for LLMs 阶段：**
- 模型采样出多条回答
- RM 评分：结构清晰的 > 混乱堆术语的
- PPO update：让高分回答的 token 轨迹更容易出现
- KL constraint：但不能为了刷分而发明怪异措辞

**对比：**
```text
SFT: "这条 gold answer 的每个 token，我都要学会"
PPO for LLMs: "这类完整回答整体更符合偏好 -> 让生成这类回答的概率提升"
              + "但不要离 SFT checkpoint 太远 -> KL constraint"
```

#### 3.6 一句话压缩（axioms version）

```text
Problem: SFT 只能 imitation，RM 可能 noisy，直接追 reward 会破坏语言能力
Axioms: (1) token-level likelihood ≠ preference optimization, (2) RM imperfect, (3) language distribution fragile
Inevitable Outcome: PPO's conservative update + KL penalty against reference distribution
Compression Mechanism: KL distance as a measure of "how far from SFT capabilities"
```

---

### 4. Verification：独立重推导——遮住答案，你能重建这个设计吗？

#### 4.1 从 axioms 出发能推导出什么？

**已知：**
- Axiom 1: SFT 只能优化 token-level likelihood → 无法直接优化回答级偏好
- Axiom 2: RM 不完美 (noisy, exploitable)
- Axiom 3: LLM 的语言能力是脆弱的，剧烈更新会破坏

**问题：** 如何让模型朝高偏好方向移动，同时不破坏语言能力？

#### 4.2 Step-by-step 推导（试着遮住答案自己推）

**Q1:** SFT 够不够？
- **不够。** 因为 SFT 只能 imitation，不能直接优化"回答级偏好"

**Q2:** 那直接用 RM + Policy Gradient 行不行？
- **不行。** 因为 RM 不完美，模型会 exploit RM → 怪异措辞、刷分、语言崩塌

**Q3:** 需要什么机制来防跑偏？
- **保守更新** — PPO 的 clip 限制单步幅度（第一层护栏）
- **分布距离约束** — KL penalty 限制整体偏离程度（第二层护栏）

**Q4:** KL constraint 为什么是"对参考分布的距离"而不是别的？
- 因为 SFT checkpoint 代表了"人类示范 + 语言能力"的平衡点
- RM 只指导方向，但不能让它完全接管策略
- **KL distance = "离原始语言能力有多远"**

**Q5:** PPO for LLMs 的本质是什么？
```text
Reward Model 说：这个方向更好
Reference Model 说：别跑太远
PPO 负责：在两者之间找到保守的更新路径
```

#### 4.3 Verification Test：如果你能回答这三个问题，就真懂了

1. **如果去掉 KL penalty 会发生什么？** — 模型会 exploit RM，语言分布崩塌
2. **为什么 PPO 的 clip 还不够？** — 只限制单步 update，不限制最终策略距离
3. **Reference model 选哪里最合适？为什么？** — SFT checkpoint，因为它有语言能力积累

#### 4.4 它是不是最终答案？

**不是。**

PPO for LLMs 仍然有很多成本：
- rollout 很贵 (采样完整回答)
- reward model 训练很贵
- value / critic head 也可能难训
- KL 系数和 reward scale 很敏感

这也是为什么后面会继续逼出更贴近 LLM 场景的变体，比如 `GRPO`。

**关键 insight：** PPO for LLMs 是"必要的妥协"——在 RM 指导和语言能力保护之间找平衡点。但这个设计本身也会暴露新的问题（成本、工程复杂度），从而继续推动新算法诞生 🔥

---

### 5. Example：同一个 prompt，SFT vs PPO for LLMs 到底在优化什么？

假设 prompt 是：
```text
请解释 Bellman 方程，并给一个最小例子。
```

#### SFT 阶段发生了什么？

**输入：** (prompt, gold_answer) pairs  
**目标：** maximize likelihood of gold answers  
**模型学到的是：** "这些 token 序列应该这样排列"

> **本质：** Imitation learning — 模仿已有示范，但不直接优化"回答质量偏好"

#### PPO for LLMs 阶段发生了什么？

1. **采样阶段：**
   - 同一个 prompt，用当前 policy π_θ 采样出多条回答（比如 4-8 条）
   - 每条回答是一个 token trajectory

2. **RM 评分阶段：**
   - 完整回答生成完后，reward model 给分
   - 回答 A：结构清楚 → R(A) = 0.85
   - 回答 B：混乱堆术语 → R(B) = 0.32

3. **PPO update 阶段：**
   ```text
   Advantage signal: A = R(A) - baseline (来自 critic / value head)
   
   Update rule: 
   maximize E[min(r_θ(a|s)*A, clip(r_θ(a|s), 1-ε, 1+ε)*A)] - β * KL(π_θ || π_ref)
   
   Where r_θ = π_θ / π_old (importance sampling ratio)
   ```

4. **KL constraint 的作用：**
   - 如果某个 token 决策在 SFT checkpoint 里概率很低，但 RM 给了高分
   - KL penalty 会惩罚这种"为了刷分而发明怪异模式"的行为
   - **本质：RM 指导方向，但不能让它完全接管策略**

#### 对比表格（压缩版）

| 维度 | SFT | PPO for LLMs |
|------|-----|--------------|
| 优化目标 | token-level likelihood | response-level preference + KL constraint |
| 信号来源 | gold answer (supervised) | reward model (preference signal) |
| 更新风格 | imitation | conservative update against reference |
| 风险 | 过拟合示范数据 | exploit RM / language distribution collapse |
| 防跑偏机制 | - | PPO clip + KL penalty |

#### 关键差异（一句话）

```text
SFT: "这些 token 序列应该这样排列" (imitation)
PPO for LLMs: "这类完整回答整体更符合偏好 -> 让生成这类回答的概率提升，但不要离 SFT checkpoint 太远"
```

---

### 6. 这一章最后压成一张因果地图（axioms -> inevitable outcome）

```text
Problem:
SFT 只能模仿示范，RM 能指导偏好但可能 noisy，直接追 reward 会破坏语言能力

Axioms (不可绕过的底层事实):
1. token-level likelihood ≠ response-level preference optimization
2. RM is imperfect (noisy, exploitable)
3. LLM's language distribution is fragile to aggressive updates

Forced Problems (如果只接受 axioms，什么矛盾会出现?):
- 只用 SFT → 无法直接优化回答级偏好
- 只用 RM + PG → exploit RM，语言崩塌
- 需要什么？→ 在 RM 指导和能力保护之间找平衡点

Inevitable Outcome:
PPO's conservative update (第一层护栏) + KL penalty against reference distribution (第二层护栏)

Compression Mechanism:
KL(π_θ || π_ref) = measure of "how far from SFT capabilities"
→ RM 指导优化方向，但不让它完全接管策略

Verification (遮住答案能重建吗?):
Q1: 去掉 KL penalty 会怎样？→ exploit RM, language collapse
Q2: PPO clip 为什么不够？→ 只限制单步 update，不限制整体分布距离  
Q3: Reference model 选哪里？→ SFT checkpoint (有语言能力积累)

Example:
SFT: "这些 token 序列应该这样排列" (imitation)
PPO for LLMs: "这类完整回答整体更符合偏好 -> 提升概率，但 KL constraint 别跑偏"
```

---

### 本章速记卡片（axioms version）

#### 一句话主线

- `LLM` = token-level policy, complete response = trajectory  
- `RM` gives preference signal at response level  
- **PPO + KL** is the *inevitable solution* to: optimize preferences without breaking language capabilities  

#### Axioms（必背底层事实）

1. **SFT limitation:** token-level likelihood ≠ preference optimization  
2. **RM limitation:** noisy, exploitable, may not align with true human preference  
3. **LLM fragility:** language distribution is cumulative (pretraining + SFT), aggressive updates break it  

#### Inevitable Outcome（为什么必须这么设计？）

- PPO's clip → 保守更新的第一层护栏（限制单步幅度）
- KL penalty → 保守更新的第二层护栏（限制整体分布距离）
- Reference model = SFT checkpoint → "语言能力锚点"

#### Verification Test（遮住答案能重建吗？）

1. **去掉 KL penalty？** — exploit RM, language collapse  
2. **PPO clip 为什么不够？** — only limits single step, not overall distribution distance  
3. **Reference model 选哪里？为什么？** — SFT checkpoint, because it has accumulated language capabilities  

#### 最短背诵版（压缩成因果链）

```text
SFT → 只能 imitation ← problem
RM + PG → exploit RM ← new problem
需要什么？→ 在 RM 指导和能力保护之间找平衡点
PPO clip → 限制单步 update (第一层)
KL penalty → 限制分布距离 (第二层)
Inevitable: PPO for LLMs = conservative update against reference distribution
```

---

> 🔥 **关键 insight:** PPO for LLMs 不是"把 PPO 搬到 LLM"，而是**在 axioms 约束下的 inevitable outcome**——RM 指导优化方向，Reference model 保护语言能力，PPO 负责保守更新。这个设计本身也会暴露新问题（成本、工程复杂度），继续推动 GRPO 等变体诞生。

---

