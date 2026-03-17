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

### 序：从 Q-Table 到 DQN

这一章如果按第一性去看，核心不是“又多了一个新算法”，而是：

> **第四章已经解决了“怎么更新 Q”，但还没解决“当 Q 根本存不下时怎么办”。**

所以第五章的推导链应该这样看：

```text
问题：Q-table 在大状态空间里爆炸
-> 出发点：Q-Learning 的 Bellman 更新没错，错的是表示方式
-> 发明：用可泛化函数 Q(s,a; theta) 代替查表
-> 验证：它是否保留了 Bellman 结构，并真的缓解了可扩展性问题
-> 示例：CartPole
```

一句话先压住：

> **DQN 不是推翻 Q-Learning，而是给 Q-Learning 换了一个能扩展的表示方式。**

---

### 1. 问题是什么：Q 表为什么会爆炸？

先从最根本的问题出发。

Q-Learning 的更新公式没有问题：

```math
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'}Q(s',a') - Q(s,a)]
```

问题不在公式，而在 **存储方式**。

Q-table 的前提是：
- 状态空间是有限的
- 每个状态都能被枚举
- 每个状态都能被唯一索引

但真实任务不是这样。

以 `CartPole` 为例，状态通常是：

```text
s = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
```

这 4 个量都是连续值。

这意味着：
- 不存在一个天然有限的小表可以全部装下
- 即使强行离散化，也会丢掉精度
- 稍微复杂一点的任务，表大小就会指数爆炸

所以本质问题是：

> **Q-Learning 缺的不是更新规则，而是可扩展的函数表示。**

---

### 2. 出发点是什么：问题不在 Bellman，而在表示方式

一个最自然的补丁是：

> 那我把连续状态硬切成格子，不就又能用表了吗？

比如：
- 角度按 100 个区间切
- 速度按 100 个区间切
- 位置按 100 个区间切
- 角速度按 100 个区间切

那状态总数就是：

```text
100^4 = 100,000,000
```

这还只是 4 维。

如果换成图像输入，状态是 `84 x 84 x 4` 像素堆叠：

```text
状态维度 = 28224
```

这时 Q-table 已经不是大一点的问题，而是根本没法用。

更致命的是，Q-table 没有泛化能力：
- 学过状态 A，不代表会状态 B
- 即使 A 和 B 很像，也得分开学

这暴露出更深的缺陷：

> **Q-table 只会记忆，不会抽象。**

---

### 3. 如何发明 DQN：用函数逼近代替查表

既然表装不下，那就别存表了。

新的想法是：

> **不要为每个状态单独存一个 Q 值，而是训练一个函数，输入状态，输出各动作的 Q 值。**

写成形式就是：

```math
Q(s,a) \approx Q(s,a; \theta)
```

其中：
- `\theta` 是神经网络参数
- 输入是状态 `s`
- 输出是每个动作的价值估计

如果动作是离散的，比如 `CartPole` 只有左/右两个动作：

```text
输入:  s = [x, x_dot, theta, theta_dot]
输出: [Q(s, left), Q(s, right)]
```

于是：
- 不需要维护巨大表格
- 相似状态可以共享参数
- 学到的规律可以泛化到没见过的新状态

这就是 DQN 的第一步突破：

> **把 Q 从“记忆表”升级成“可泛化函数”。**

---

### 4. 如何让这个发明能工作？

#### 4.1 核心结构

DQN 仍然在学 Q，只是把表换成网络：

```text
状态 s
  |
  v
Neural Network Q(s; theta)
  |
  +--> Q(s, left)
  +--> Q(s, right)
```

然后动作选择还是和以前一样：

```math
a = \arg\max_a Q(s,a; \theta)
```

也就是说：
- **决策逻辑没变**：还是选 Q 最大的动作
- **Bellman target 没变**：还是 `r + gamma max Q(s',a')`
- **变化的只是 Q 的表示方式**

---

#### 4.2 训练目标

DQN 的训练本质是：

> **让网络输出的 Q 值，逼近 Bellman target。**

目标写成：

```math
y = r + \gamma \max_{a'} Q(s',a'; \theta^-)
```

损失函数：

```math
L(\theta) = (y - Q(s,a; \theta))^2
```

这就是一个监督学习味很重的过程：
- 输入：状态 `s`
- 预测：`Q(s,a; \theta)`
- 标签：Bellman target `y`
- 优化：最小化 MSE

但要注意：

> **这个标签不是外部真值，而是 RL 自己构造出来的 bootstrap target。**

这就是 DQN 的特别之处。

---

#### 4.3 为什么要有 Replay Buffer？

如果你每一步交互完就立刻拿当前样本训练，会有两个问题：

1. 相邻样本太像，训练数据强相关
2. 网络刚更新，target 又跟着变，系统容易抖

所以 DQN 引入了 **经验回放（Experience Replay）**：

```text
(s, a, r, s', done)
```

都先丢进一个 buffer 里。

训练时从里面随机采样 batch：

```python
batch = random.sample(replay_buffer, batch_size)
```

效果：
- 打散样本相关性
- 提高样本利用率
- 让训练更像稳定的 i.i.d. 学习

---

#### 4.4 为什么要有 Target Network？

如果你直接用同一个网络同时算：
- 当前 Q
- 下一状态 target

那就会出现：

> **你一边改答案，一边又拿改动中的答案当老师。**

这很容易发散。

所以 DQN 会维护两套网络：

- `online network`: 当前正在训练的网络 `Q(s,a; \theta)`
- `target network`: 延迟更新的目标网络 `Q(s,a; \theta^-)`

更新流程：

```python
if step % target_update == 0:
    target_net.load_state_dict(online_net.state_dict())
```

直觉上：
- online net 负责学
- target net 负责暂时当“较稳定的老师”

所以 DQN 的稳定性，靠的是两个补丁：
- replay buffer
- target network

---

### 5. 最小示例：用 DQN 学 CartPole

#### 5.1 任务是什么？

`CartPole` 的目标是：

> **控制小车左右移动，让杆子尽量不要倒。**

状态：

```text
[x, x_dot, theta, theta_dot]
```

动作：
- `0`: left
- `1`: right

奖励：
- 每坚持 1 步，奖励 `+1`

为什么这个任务适合 DQN？
因为：
- 动作是离散的（左/右）
- 状态是连续的（Q-table 不适合）
- 刚好体现“表不够用，网络接管”的过渡

---

#### 5.2 训练逻辑

```python
for episode in range(num_episodes):
    state = env.reset()

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = online_net(state)
            action = argmax(q_values)

        next_state, reward, done = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            train_dqn(batch)
```

`train_dqn(batch)` 核心：

```python
y = r + gamma * max(target_net(next_state))
loss = mse(online_net(state)[action], y)
backward(loss)
```

随着训练推进：
- 网络逐渐学会什么姿态下该往左，什么姿态下该往右
- 平均坚持步数会从十几步涨到几百步

---

### 6. Role：DQN 在 RL 发展史里的角色

DQN 的地位非常关键，因为它完成了一个历史级跨越：

> **把 value-based RL 从“小表格玩具”，推进到了“高维感知任务”。**

它让大家第一次真正看到：
- RL 可以吃连续状态
- 神经网络可以学控制策略
- Atari 这种像素级输入也能直接做决策

一句话概括：

> **Q-Learning 解决“怎么学动作价值”，DQN 解决“在大状态空间里怎么表示动作价值”。**

---

### 7. Effect：DQN 带来了什么改变？

1. **从记忆走向泛化**
   - Q-table: 见过才会
   - DQN: 没见过但相似，也能猜个八九不离十

2. **从小状态空间走向高维输入**
   - 可以处理连续状态
   - 可以接图像、传感器、向量特征

3. **把深度学习正式接入 RL**
   - 这条线后来长出了 Double DQN、Dueling DQN、Rainbow 等一堆变体

但也要看到它的边界：
- 主要适合离散动作空间
- 训练仍可能不稳定
- 样本效率并不算特别高

所以接下来就会自然出现一个新方向：

> **既然我最终想要的是策略，能不能别绕道学 Q，直接优化策略本身？**

这就自然过渡到下一章：`Policy Gradient`。

---

### 本章速记卡片

#### 一句话主线

- Q-Learning 的思想没问题，问题是 Q-table 装不下大状态空间
- DQN 用神经网络 `Q(s,a; \theta)` 代替 Q-table
- 训练目标仍然来自 Bellman target
- 为了稳定训练，需要 replay buffer 和 target network

#### 必背公式

```math
y = r + \gamma \max_{a'} Q(s',a'; \theta^-)
```

```math
L(\theta) = (y - Q(s,a; \theta))^2
```

#### 必背术语

- `Function Approximation`：函数逼近
- `Replay Buffer`：经验回放池
- `Target Network`：目标网络
- `Online Network`：在线训练网络

#### 最短背诵版

1. Q-table 不可扩展
2. DQN 用网络输出 Q 值
3. Bellman target 仍然是老师
4. Replay Buffer 打散样本相关性
5. Target Network 防止训练目标乱飘

---

