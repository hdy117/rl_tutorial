## 第十二章：GRPO - 在 LLM 场景里，为什么“组内相对比较”会自然替代一部分 critic 角色？

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

> 你在这里：LLM 分支 -> GRPO

### 序：GRPO 不是凭空冒出来的新名字，而是 LLM 场景继续把 PPO 改造下去的结果

第十一章已经把一件事讲清楚了：

> LLM 可以被看成 policy，PPO 可以接进来做保守优化。

但新的问题也同时出现：
- rollout 贵
- critic / value 训练复杂
- 回答质量很多时候本来就是相对比较出来的

这就会继续逼出一个更贴近 LLM 反馈结构的问题：

> **既然同一个 prompt 下我们本来就会采多条回答并做比较，那能不能直接利用“组内相对优劣”来构造策略更新信号？**

一句话先压住：

> **GRPO 的本质，是保留 PPO 的保守更新思想，但把 advantage 的构造更多转移到同 prompt 多样本的组内相对奖励上。**

---

### 1. Problem：为什么 PPO for LLMs 之后还会继续逼出 GRPO？

#### 1.1 LLM 的反馈天然就是“同题比答案”

很多 LLM 任务里，最自然的数据形态本来就是：
- 同一个 prompt
- 采样多条 responses
- 比较谁更好

所以如果还完全依赖 critic 去估一个绝对 value，会显得有点绕。

#### 1.2 critic 在大模型后训练里很贵

如果你还要配：
- value head
- GAE
- 额外稳定训练逻辑

那工程复杂度会明显上升。

所以 LLM 场景会自然追问：

> **能不能更多利用“同组比较”本身来构造 baseline，而不是重度依赖单独 critic？**

#### 1.3 回答质量很多时候本来就只需要相对信号

对一个 prompt，4 条回答里：
- 谁最好
- 谁最差
- 谁比谁略好

这些相对信息往往已经足够驱动策略更新。

也就是说，GRPO 面对的问题不是“PPO 错了”，而是：

> **PPO 在 LLM 里还能不能更贴合真实反馈形态、同时更省掉一部分 value 建模负担？**

---

### 2. Starting Point：如果同组里能直接比较，那 baseline 就不一定非要来自 critic

从第一性原理看，策略更新真正需要的是：

> **这个样本，相对 baseline 来说，是更值得强化，还是更该削弱？**

在 Actor-Critic / PPO 里，这个 baseline 常常来自：
- value function
- advantage estimate
- TD / GAE 等结构

但在 LLM 同 prompt 多采样的场景里，一个更自然的 baseline 是：
- 同组其他回答的平均水平
- 同组中的相对排名
- 组内标准化后的 reward

所以这个出发点很关键：

```text
只要能构造“相对平均更好还是更差”的信号
baseline 不一定非要由 critic 单独学习出来
```

这就是 `group-relative` 这三个字真正有力量的地方。

---

### 3. Invention：GRPO 到底改了 PPO 的哪一层？

#### 3.1 训练单元从“单条 response”变成“同 prompt 下的一组 responses”

GRPO 里更自然的样本组织方式是：
- 固定一个 prompt
- 采样 `k` 条回答
- 对这 `k` 条回答打分
- 在组内做相对比较

于是策略更新不再只看：
- 这一条回答绝对得了多少分

而更看：
- 它在这一组里相对更好多少

#### 3.2 relative advantage 被组内比较逼出来

如果一组 reward 是：

```text
[0.9, 0.7, 0.2, -0.1]
```

那最自然的问题不是：
- 第一个样本值多少钱

而是：
- 第一个样本比这一组平均高多少
- 第四个样本比平均低多少

于是你就会得到某种组内相对优势信号：
- 高于组均值 -> 强化
- 低于组均值 -> 削弱

这实际上是在用组内结构充当 baseline。

#### 3.3 PPO 的保守更新思想仍然保留

这一点别丢。

GRPO 不是说：
- 有了组内比较，就可以随便猛更了

恰恰相反，它仍然继承：
- 新旧策略不能一下差太远
- 更新要保守
- 防止模型因为局部噪声突然跑偏

所以最简压缩是：

```text
GRPO = PPO 的保守更新框架 + 组内相对优势信号
```

#### 3.4 为什么它特别适合 reasoning / multi-sample 场景？

因为很多推理任务本来就适合：
- 同题多采样
- 比较答案对错、过程质量、格式满足度

这时组内相对信息密度很高，能直接拿来驱动优化。

---

### 4. Verification：为什么这种改造在 LLM 场景里是自然的？

#### 4.1 它有没有更贴合真实数据形态？

有。

因为 LLM 后训练里，人类或 reward pipeline 很多时候本来就在做：
- 多回答比较
- 排序
- 胜负判断

GRPO 只是把这种数据结构更直接地变成优化信号。

#### 4.2 它有没有减轻一部分 critic 负担？

方向上有。

因为 baseline 的一部分功能，现在由：
- 组内平均
- 相对排名
- 组内标准化奖励

来承担。

这能让方法更贴近“比较式反馈”这个现实。

#### 4.3 它是不是完全替代 PPO？

不是。

更准确地说：
- 它建立在 PPO 的保守更新思想之上
- 只是把 advantage 构造改得更 LLM-native

所以理解关系时一定要说：

> **GRPO 不是推翻 PPO，而是 PPO 在 LLM 多样本相对奖励场景下的一种自然特化。**

---

### 5. Example：同一个数学题，多条推理答案如何驱动 GRPO？

prompt：

```text
解一个二次方程，并写出推导过程。
```

模型采样 4 条回答：
- A：答案对，过程清楚
- B：答案对，但过程乱
- C：过程看似合理，但中间算错
- D：答案和过程都不对

如果组内 reward 大致是：

```text
A: 0.95
B: 0.75
C: 0.10
D: -0.20
```

那 GRPO 的直觉就是：
- A/B 这类回答高于组平均 -> 强化
- C/D 低于组平均 -> 削弱
- 同时仍然限制新策略别一步走太猛

也就是说，它直接利用了“同题比较”这件事本身。

---

### 6. 这一章最后压成一张脑图

```text
Problem:
PPO for LLMs 依然可能依赖较重的 critic/value 结构，而 LLM 反馈天然更像同组比较

Starting Point:
策略更新真正需要的是相对 baseline 信号，而 baseline 不一定非要来自独立 critic

Invention:
同 prompt 多采样 -> 组内比较 -> 构造 relative advantage
再配合 PPO 风格的保守更新

Verification:
它更贴合 LLM 后训练的比较式反馈结构
同时保留了“别走太猛”的核心思想

Example:
同一道数学题下，多条回答按组内相对质量驱动策略更新
```

### 本章速记卡片

#### 一句话主线

- `GRPO` 建立在 `PPO` 上
- 它不再那么依赖单独 critic 来给 baseline
- 而是更多利用同 prompt 多样本的组内相对奖励
- 所以它特别适合 `LLM reasoning post-training`

#### 必背术语

- `Group`
- `Relative Reward`
- `Relative Advantage`
- `Baseline`
- `GRPO`

#### 最短背诵版

1. 同一个 prompt 采多条回答
2. 组内比较谁更好
3. 高于组平均的强化，低于组平均的削弱
4. 同时保留 PPO 的保守更新
5. 这就是 GRPO 的核心直觉

---

