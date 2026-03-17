## 第九章：LLM 分支入口 - 为什么 GRPO 应该放在 PPO 之后？

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

> 你在这里：LLM 分支入口 -> 为什么从 PPO 转向 RLHF / GRPO 语境

### 序：GRPO 不是“经典控制主线的下一站”，而是“PPO 在 LLM 场景下的变体”

先把位置说死，不然后面章节会越写越乱：

> **GRPO 最合适放在 PPO 之后，但不要把它当成 DQN / Actor-Critic 那种通用 RL 主线算法，而要把它放进“LLM post-training / RL for language models”这条分支里。**

原因很简单：
- `PPO` 解决的是：策略更新别太猛
- `GRPO` 继承的正是这条思路
- 但它服务的主要场景，不是经典控制，而是 **LLM 对齐 / reasoning post-training**

所以它在教程里的位置，最自然应该是：

```text
经典 RL 主线：
Q-Learning -> DQN -> Policy Gradient -> Actor-Critic -> PPO

然后分叉：
- 控制 / 连续动作分支：DDPG / TD3 / SAC
- LLM 分支：RLHF / PPO for LLMs -> GRPO
```

一句话先压住：

> **GRPO 不该插在 PPO 前面，也不该硬塞进经典机器人控制主线；它最适合放在“PPO 之后，作为 LLM 强化学习特化变体”来讲。**

---

### 1. Problem：为什么不能把 GRPO 和 DQN / PPO 并排当成同一级主线算法？

因为它们解决的问题层级不一样。

#### 1.1 经典 RL 主线关心的是“如何学会行动”

前面这些章节主要在解决：
- 怎么定义 value
- 怎么从采样中更新策略
- 怎么降低方差
- 怎么稳定策略更新
- 怎么处理连续动作

这些问题是 **通用 RL** 问题。

不管你是在：
- 走迷宫
- 控机器人
- 玩 Atari

都会遇到。

#### 1.2 GRPO 关心的是“LLM 场景下，怎么更便宜地做相对策略优化”

GRPO 的语境明显更窄：
- 一个 prompt 往往会采样多条回答
- 奖励很多时候是相对比较出来的，而不是环境一步一步给的 dense reward
- 训练对象是 language model，不是传统 control policy
- 我们常常更关心生成质量排序、group-relative 信号、以及减少额外 value model 成本

所以从“问题被什么逼出来”这个角度看，GRPO 并不是对前面所有 RL 任务都自然适用的下一站，而是：

> **在 PPO 已经成立之后，LLM 训练场景又提出了新要求，于是长出来的专用分支。**

---

### 2. Starting Point：如果 PPO 已经能做策略优化，LLM 场景还缺什么？

从第一性原理问：

> **LLM 后训练里的反馈，和机器人控制里的反馈，到底有什么不同？**

差别非常大。

#### 2.1 LLM 常常不是“每一步都有环境奖励”

在机器人里，你可以有：
- 每走一步的 reward
- 摔倒的惩罚
- 能耗惩罚

但在 LLM 里，很多时候更像这样：
- 给一个 prompt
- 采样几条完整回答
- 用 reward model 或规则给整条回答打分
- 再比较这些回答谁更好

也就是说，反馈更像：
- **序列级 / 回答级**
- **相对排序式**
- 而不是传统 control 里的逐步环境回报

#### 2.2 如果直接照搬 PPO，value / critic 这一套在 LLM 里成本很高

PPO 在很多实现里会搭配 value function / critic：
- 用来估 advantage
- 用来降低方差

但放到大语言模型后训练里，代价会很重：
- 训练更复杂
- 需要额外 value head 或 value model
- 稳定性和工程成本都上去

于是 LLM 场景会自然提出一个新问题：

> **能不能保留 PPO 这种“限制策略别一步改太猛”的优点，同时又更贴合“多答案比较”这种反馈形式，并尽量减少 critic 负担？**

这就是 GRPO 的出发点。

---

### 3. Invention：GRPO 是怎么从 PPO 的语境里长出来的？

先说本质，不先堆细节：

> **GRPO 可以看成是：把 PPO 的“保守策略更新”思想，和 LLM 场景里的“组内相对奖励”结合起来。**

#### 3.1 从“单条样本值多少钱”转向“同组里谁相对更好”

在 LLM 场景里，一个很自然的训练单元是：
- 同一个 prompt
- 采样出一组 responses
- 对这组 responses 打分或比较

这时候最有信息量的问题常常不是：

```text
这条回答的绝对价值是多少？
```

而是：

```text
在同一个 prompt 的这组回答里，哪条更好？好多少？
```

这就天然引出“group-relative”信号。

#### 3.2 Advantage 的来源不再主要靠 critic，而更像组内相对基线

PPO / Actor-Critic 里，`Advantage` 常常来自：
- value estimate
- GAE
- TD-style baseline

但在 GRPO 的语境里，更自然的是：
- 同一个 prompt 下采多条样本
- 比较它们的 reward
- 用组内均值、相对排名、标准化分数之类的方式形成相对优势信号

也就是说，它把“这条样本比 baseline 好多少”这个问题，更多地交给：
- **组内相对比较**

而不是：
- 单独训练一个 critic 去估每个 token / 序列的 value

#### 3.3 PPO 的“别走太猛”思想仍然保留

这点很关键。

如果没有 PPO 那条主线，你很难理解 GRPO 的核心节制逻辑。

GRPO 并不是说：
- 我们不要保守更新了

而是说：
- **仍然需要限制新旧策略差异，避免一步把模型带偏**
- 只不过 advantage / reward 这部分，更偏向组内相对化构造

所以它和 PPO 的关系更像：

```text
PPO:  保守策略更新 + advantage
GRPO: 保守策略更新 + group-relative advantage
```

这就是为什么它一定要建立在你已经理解 PPO 之后再讲。

---

### 4. Verification：为什么说“PPO 之后讲 GRPO”比“把 GRPO 提前”更合理？

#### 4.1 不先讲 PPO，你很难解释 GRPO 到底继承了什么

如果学生还没理解：
- 为什么 policy update 会过猛
- 为什么要限制新旧策略差异
- surrogate objective 在干什么

那你一上来讲 GRPO，很容易变成：
- 记一个名字
- 背一个实现
- 知道它“好像是 RLHF / DeepSeek 在用的”

但不知道它到底是从什么矛盾里长出来的。

#### 4.2 不先切到 LLM 场景，你也很难解释它为什么不是通用主线算法

如果不先告诉读者：
- 经典控制任务和 LLM 后训练的反馈结构不一样

那 GRPO 就会看起来像：
- “PPO 的升级版”
- 或者“下一个更高级的标准算法”

这其实会误导。

更准确的理解是：

> **GRPO 不是对 PPO 的普遍替代，而是 PPO 思想在 LLM 组相对奖励场景下的一种特化实现。**

#### 4.3 所以最自然的教学顺序是什么？

最顺的顺序是：

```text
Policy Gradient
-> Actor-Critic
-> PPO
-> RLHF / LLM post-training 背景
-> GRPO
```

这样每一步都在回答：
- 前一步留下了什么问题
- 新场景又提出了什么限制
- 新方法为什么非长出来不可

这条因果线就会非常顺。

---

### 5. Example：把 PPO 和 GRPO 放在 LLM 场景里对比看

假设现在有一个数学推理 prompt：

```text
题目：求一个方程的解
```

模型针对同一个 prompt 采样出 4 个回答：
- 回答 A：过程完整，答案对
- 回答 B：答案对，但过程乱
- 回答 C：过程像样，但算错
- 回答 D：胡说八道

#### 5.1 如果按经典 PPO 视角看

你会想：
- 每条回答有一个 reward
- 再结合 value / advantage 去做保守策略更新

这当然能做，但在 LLM 场景里会显得比较重。

#### 5.2 如果按 GRPO 视角看

你会更自然地想：
- 这 4 条是同组 samples
- 我更关心它们的相对好坏
- A 明显优于 D，B 略优于 C
- 所以更新时应该让模型更偏向 A/B 这类回答风格，远离 C/D

也就是说，信号的构造方式会更像：

```text
同 prompt 下，多样本组内比较
-> 得到相对优势
-> 再做保守策略更新
```

这就是它为什么特别适合放在 LLM post-training 那一支里。

---

### 6. 教程结构上，最推荐怎么放？

如果你的教程目标是“先打通通用 RL 主线，再进入 LLM RL”，那我推荐这样排：

```text
第 1-8 章：经典 RL 主线
1. RL 基本元素
2. Policy
3. Value / Bellman
4. Q-Learning
5. DQN
6. Policy Gradient
7. Actor-Critic
8. PPO

第 9 章：从经典 RL 到 LLM post-training
- 为什么 LLM 的反馈结构不同
- RLHF 在解决什么问题
- 为什么 PPO 会先被拿来做 LLM 对齐

第 10 章：GRPO
- 为什么在 LLM 场景下，需要 group-relative 信号
- 它和 PPO 的继承关系是什么
- 它解决了什么工程 / 稳定性 / 成本问题
```

如果你暂时不想单开一章讲 RLHF，也可以把它收成：

```text
第九章：LLM 强化学习导论（含 PPO -> GRPO 过渡）
```

但无论如何，**不要把 GRPO 放到 PPO 前面，也不要放到 DDPG / TD3 / SAC 那条连续控制支线中间。**

---

### 7. 这一章最后压成一张脑图

```text
Problem:
GRPO 不是通用 RL 主线算法，直接塞进经典控制路径会让结构混乱

Starting Point:
LLM post-training 的反馈更像组内相对比较，而不是传统环境逐步 reward

Invention:
保留 PPO 的保守更新思想
把 advantage 更多建立在 group-relative reward / baseline 上

Verification:
只有先懂 PPO，再切到 LLM 场景，GRPO 的位置和意义才会自然

Example:
同一个 prompt 下采多条回答，按组内相对质量更新策略
```

### 本章速记卡片

#### 一句话主线

- GRPO 最适合放在 `PPO` 后面讲
- 但它属于 `LLM post-training` 分支，不是经典控制主线必经站
- 它继承了 PPO 的保守更新思想
- 同时把优势信号更多建立在组内相对比较上

#### 必背判断

- `PPO`：通用的保守策略优化主线方法
- `GRPO`：PPO 思想在 LLM 组相对奖励场景下的特化分支
- `不要` 放在 PPO 前面
- `不要` 硬塞进经典连续控制支线

#### 最短背诵版

1. 先学 PPO
2. 再切到 LLM post-training
3. 然后讲 GRPO
4. 因为 GRPO 依赖 PPO 的保守更新思想
5. 但它的真正语境是语言模型组相对奖励

---

