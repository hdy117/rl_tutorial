## 第五章：Deep Q-Network (DQN) - 用神经网络替代 Q 表

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

> 你在这里：主干 -> DQN

### 学习脊柱 (Learning Spine)

这一章不是讲"又多了一个新算法"，而是解决 Q-Learning 在真实世界里的**可扩展性瓶颈**。

```text
问题 → 出发点 → 发明 → 验证 → 示例
```

---

## Problem：什么矛盾迫使 DQN 必须存在？

Q-Learning 的更新公式本身没有缺陷：

```math
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'}Q(s',a') - Q(s,a)]
```

但**存储方式**彻底失效了。

### Q-table 的三个致命限制

1. **状态空间必须有限且可枚举**
   ```text
   CartPole: s = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
   → 4 个连续变量，无法直接查表
   ```

2. **离散化会指数爆炸**
   - 每个维度切 100 格：`100^4 = 100,000,000` 个状态
   - 稍微复杂一点的任务就不可行

3. **没有泛化能力**
   - 学过状态 A，不代表会状态 B
   - 即使 A 和 B 几乎一样，也得分开学一遍

> **核心矛盾：Q-Learning 的更新规则完美，但 Q-table 无法扩展到真实任务。**

---

## Starting Point：我们已有的工具是什么？

### 已确认的事实

1. **Bellman 方程是正确的** - 它定义了 RL 问题的本质结构
2. **Q-Learning 的更新逻辑是对的** - `r + γ max Q(s',a')` 作为 target 有效
3. **问题是表示，不是算法** - 如果把 Q-table 换成别的存储方式会怎样？

### 唯一可行的路径

如果状态空间太大无法查表，那么：

> **必须用一个能泛化的函数来代替表格。**

这不是"新想法"，而是**被迫的唯一选择**。

---

## Invention：如何推导出 DQN？

### 核心发明：Q(s,a) → Q(s,a; θ)

不是把 Q-table 改大一点，而是彻底换掉存储方式：

```text
旧方式：Q[s][a] = float (查表)
新方式：Q(s, a; θ) = neural_network_output (函数逼近)
```

**关键洞察：**
- 输入状态 `s` → 神经网络输出各动作的 Q 值
- 参数 θ 被训练成"压缩版的 Q-table"
- 相似状态共享参数 → 自动泛化

### CartPole 的例子

```text
输入：s = [x, x_dot, theta, theta_dot] (4 个连续值)
输出：[Q(s, left), Q(s, right)] (2 个动作的 Q 值)
```

不需要离散步骤，直接端到端。

---

## Verification：如何证明这个方案真的有效？

### 验证标准 1：保留 Bellman 结构

DQN 的训练目标仍然是：

```math
y = r + \gamma \max_{a'} Q(s',a'; \theta^-)
L(\theta) = (y - Q(s,a; \theta))^2
```

**关键：** target 仍然是 bootstrap 的 Bellman target，只是 Q 现在是函数而不是表格。

### 验证标准 2：解决可扩展性问题

对比实验（理论推导）：

| 场景 | Q-table | DQN |
|------|---------|-----|
| CartPole (连续状态) | ❌ 无法直接用 | ✅ 直接处理 |
| Atari (像素输入) | ❌ 完全不可能 | ✅ 端到端学习 |
| 泛化到未见状态 | ❌ 不会 | ✅ 会 |

### 验证标准 3：训练稳定性补丁

**问题：** 直接用神经网络学 Q-Learning 会发散，为什么？

1. **样本强相关** - 连续步骤的状态几乎一样
2. **目标漂移** - target 网络自己也在变，导致训练目标乱飘

**补丁（不是核心发明）：**

- `Replay Buffer`：打散相关性，让数据更接近 i.i.d.
- `Target Network`：固定 teacher，防止"自己改答案又拿新答案当老师"

> **验证结论：DQN = Q-Learning 的更新规则 + 神经网络的泛化能力。稳定性补丁是工程必要性，不是核心创新。**

---

## Example：最小可行示例 - CartPole

### 为什么选 CartPole？

这是**最小的能体现 DQN 必要性的任务**：
- ✅ 状态连续（Q-table 无法直接用）
- ✅ 动作离散（DQN 适用，还没到连续动作的复杂度）
- ❌ 不是 Cloning（那是 DDPG/TD3/SAC 的地盘）

### 训练流程骨架

```python
for episode in range(num_episodes):
    state = env.reset()
    
    # 1. 收集经验 → Replay Buffer
    while not done:
        action = epsilon_greedy_policy(state, online_net)
        next_state, reward, done = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        
    # 2. 从 buffer 采样训练 → 打散相关性
    batch = replay_buffer.sample(batch_size)
    
    # 3. Bellman target（用 target network）
    y = r + gamma * max(target_net(next_state))
    
    # 4. Loss & Backprop
    loss = mse(online_net(state)[action], y)
```

### 预期效果

- **初期**：平均坚持步数 ~10-20（随机策略水平）
- **训练后**：平均步数涨到几百（学会平衡杆子）
- **泛化验证**：给没见过的新初始状态，也能正确应对

---

## Compression：DQN 的本质压缩形式

### 最简表达

```text
Q-Learning + Function Approximation = DQN
Replay Buffer + Target Network = 稳定性补丁（工程必要性）
```

### 历史定位

> **DQN 完成了价值函数 RL 从"小表格玩具"到"高维感知任务"的跨越。**

- Q-Learning：解决了"**怎么学动作价值**"
- DQN：解决了"**在大状态空间里怎么表示动作价值**"

---

## Transition：接下来去哪里？

### DQN 的边界

1. **只能处理离散动作** - CartPole（左右）、Atari（按键）OK，但连续控制不行
2. **样本效率仍然不高** - 需要大量交互才能收敛
3. **训练不稳定** - 即使有 replay buffer + target network，仍是工程挑战

### 自然的下一步

> **"既然我最终想要的是策略 π(a|s)，能不能别绕道学 Q，直接优化策略本身？"**

这就逼出了下一章：`Policy Gradient`。

---

## 本章速记卡片 (Recap Card)

### Problem
- Q-table 无法扩展到连续状态空间
- Bellman 更新规则正确，但存储方式失效

### Starting Point
- Q-Learning 的 Bellman target 已经有效
- 唯一瓶颈是表示方式的泛化能力

### Invention
- `Q(s,a) → Q(s,a; θ)`：用神经网络代替查表
- 参数共享实现自动泛化

### Verification
- Bellman 结构完全保留
- Replay Buffer + Target Network 解决稳定性
- CartPole/Atari 实验验证可扩展性

### Example (最小可行案例)
```python
# DQN 核心训练循环
batch = replay_buffer.sample(batch_size)
y = r + gamma * max(target_net(next_state))
loss = mse(online_net(state)[action], y)
backward(loss)
```

---

## 下一章预告：Policy Gradient

既然 Q-Learning 和 DQN 都是学"动作价值"，那有没有可能**直接优化策略本身**？

这就是 Policy Gradient 要解决的问题。