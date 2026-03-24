## 第七章：Actor-Critic - 结合 Policy 和 Value

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

> 你在这里：主干 -> Actor-Critic

### 序：从"直接学策略"到"给策略配一个评价器"

第六章已经把一件事说透了：

> 直接优化策略这条路，本身是对的。

但它马上暴露出一个新问题：

> **策略虽然直接学了，可更新信号太吵。**

先把这一章的主线钉住：

```text
problem -> starting point -> invention -> verification -> example
```

一句话先压住：

> **Actor-Critic 的本质，是让一个模块负责行动，让另一个模块负责评价，从而把"直接学策略"和"稳定给反馈"结合起来。**

---

### 1. Problem：纯 Policy Gradient 到底卡在哪里？

在 `REINFORCE` 里，策略更新大致长这样：

```math
G_t \nabla_\theta \log \pi_\theta(a_t|s_t)
```

问题不是方向错，而是信号太粗。

#### 1.1 每一步动作都背着整条轨迹的锅

`G_t` 是从时刻 `t` 往后的累计回报。

这意味着：
- 你在第 `t` 步做了一个动作
- 但给你的评价，掺杂了后面很多步的运气和失误

于是 credit assignment 会变得很糊：
- 到底是我刚才那一步好
- 还是后面几步运气好
- 或者只是环境随机性在作怪

#### 1.2 方差大，训练容易飘

在早期训练里尤其明显：
- 有时一条轨迹突然成功了
- 有时一条轨迹突然全崩了

如果直接用整段回报去更新，那么策略就会被这些大波动拉来拉去。

也就是说，Policy Gradient 现在遇到的不是"目标错了"，而是：

> **缺少一个更局部、更稳定的评价信号。**

---

### 2. Starting Point：Actor 需要的不是总分，而是"这一步比平均好多少"

如果从第一性原理继续推，我们要先问：

> **一个负责行动的策略模块，真正需要什么样的反馈？**

答案不是"整局最终总分"。

因为对当前动作来说，更有用的问题其实是：

> **我刚才这个动作，在当前状态下，相比平均水平到底是更好还是更差？**

这句话非常关键，因为它把反馈需求从：
- 全局、粗糙的 `G_t`

改成了：
- 局部、相对的"是否优于 baseline"

于是自然就会出现模块分工。

#### Actor
负责行动：

```math
\pi_\theta(a|s)
```

它回答的是：

> 在状态 `s` 下，我该怎么做？

#### Critic
负责评价：

```math
V_w(s) \quad 或 \quad Q_w(s,a)
```

它回答的是：

> 当前状态值多少？刚才那个动作相对平均水平怎么样？

所以这一章的出发点不是"我要两个网络"，而是：

> **我需要把"做动作"和"评估动作"分成两个职能。**

---

### 3. Invention：Actor-Critic 是怎么长出来的？

一旦接受"行动者需要评价者"，下一步自然会问：

> **Critic 到底应该给 Actor 什么信息，才最有用？**

#### 3.1 只给 `V(s)` 还不够

如果 Critic 只告诉你：

```math
V(s)
```

那意思只是：
- 这个状态整体平均值不错/不好

但 Actor 真正关心的是：
- 我刚才选的这个动作，相对于这个状态下的平均动作，到底更好还是更差？

所以只知道"状态总体值"还不够，必须有一个"相对超额收益"的量。

#### 3.2 Advantage 被逼出来了

最自然的桥梁就是：

```math
A(s,a) = Q(s,a) - V(s)
```

它的意思是：
- `Q(s,a)`：做这个具体动作的价值
- `V(s)`：这个状态下平均来说的价值
- 二者相减：这个动作比平均水平高出多少

所以：
- `A > 0` -> 这个动作比平均好，应该更常做
- `A < 0` -> 这个动作比平均差，应该少做

这一步极其关键，因为它把问题从：

```text
这局总分怎么样？
```

转成了：

```text
这一步相对平均操作，超额表现是多少？
```

#### 3.3 TD error 成为一个便宜又有效的 advantage 近似

但精确求 `Q(s,a)` 往往不便宜，所以工程上会进一步压缩成一步 TD 形式：

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

这其实就是：
- 新证据给出的 Bellman target
- 减去旧估计

于是 `\delta_t` 同时扮演两个角色：
- 对 Critic 来说，它是 TD error
- 对 Actor 来说，它可以近似"这一步比平均好多少"

这就是 Actor-Critic 最漂亮的地方：

> **Bellman 的局部递推，被重新接回了策略优化。**

#### 3.4 两个模块开始互相配合

Critic 更新 value，让它更接近真实状态值：

```math
L_{critic} = (r + \gamma V(s') - V(s))^2
```

Actor 用 advantage 信号调整策略：

```math
\nabla_\theta J(\theta) \approx \mathbb{E}[A_t \nabla_\theta \log \pi_\theta(a_t|s_t)]
```

核心直觉就一句：

> Critic 把"这一步表现怎样"翻译成稳定信号，Actor 根据这个信号调整行为。

---

### 4. Verification：怎么确认 Actor-Critic 真解决了前一章的问题？

#### 4.1 它有没有保留"直接学策略"的优点？

有。

Actor 仍然在直接优化：

```math
\pi_\theta(a|s)
```

所以：
- 连续动作仍然友好
- 策略分布仍然可以直接建模

也就是说，第六章最珍贵的东西没有丢。

#### 4.2 它有没有让更新信号更局部、更稳定？

有。

相比 `G_t` 要背整条轨迹，`A_t` 或 `\delta_t` 只关心：
- 当前 reward
- 下一个状态的价值估计
- 当前状态的旧估计

所以反馈变得更短、更局部，方差通常明显下降。

#### 4.3 它还能不能区分"比平均好"还是"比平均差"？

能。

这正是 advantage 的意义：
- 正值 -> 强化这个动作
- 负值 -> 削弱这个动作

所以 Actor 不再是被"整局总分"粗暴驱动，而是在接受更细粒度的相对反馈。

#### 4.4 它有没有新的代价？

有。

你现在不是只训练一个策略网络，而是要让：
- Actor
- Critic

同时学习、互相配合。

如果 Critic 评得不准，Actor 也会被带偏。
所以它解决了方差问题，但引入了系统耦合和实现复杂度。

验证结论可以压成一句：

> **Actor-Critic 用额外的评价模块，换来了更稳定的策略学习信号。**

---

### 5. Advantage 的方差分析：为什么要减去 baseline？ 🔥

在深入算法之前，我们先从**数学直观**上搞清楚一件事：

> **为什么减去 `V(s)` 作为 baseline，能降低策略梯度的方差？**

这一步非常关键，因为它直接说明了 Actor-Critic 的核心价值。

#### 5.1 策略梯度公式回顾

REINFORCE 的更新方向是：

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [G_t \nabla_\theta \log \pi_\theta(a_t|s_t)]
```

问题在于 `G_t` 波动太大。

#### 5.2 Baseline 可以随便减吗？

我们试试在 `G_t` 里减去一个只依赖于状态的值 `b(s)`：

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [(G_t - b(s)) \nabla_\theta \log \pi_\theta(a_t|s_t)] + \mathbb{E}_{\pi_\theta} [b(s) \nabla_\theta \log \pi_\theta(a_t|s_t)]
```

关键问题来了：**后面那项会不会破坏梯度方向？**

我们展开看看：

```math
\begin{aligned}
\mathbb{E}_{\pi_\theta} [b(s) \nabla_\theta \log \pi_\theta(a_t|s_t)] 
&= \sum_a \int \pi_\theta(a|s) b(s) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} da \\
&= b(s) \sum_a \int \nabla_\theta \pi_\theta(a|s) da \\
&= b(s) \nabla_\theta \left(\sum_a \int \pi_\theta(a|s) da\right) \\
&= b(s) \nabla_\theta (1) = 0
\end{aligned}
```

**结论：**

> **任何只依赖于状态 `b(s)` 的 baseline，都可以安全地从回报中减去，不会改变期望的梯度方向。**

但方差会变化！

#### 5.3 方差最小化的直观理解

用更直观的方式看：

假设某个状态下，平均来说 `V(s) ≈ 10`。
- 如果某次轨迹回报是 `G_t = 20`，那这个动作比平均好很多
- 如果某次轨迹回报是 `G_t = -5`，那这个动作比平均差

但如果直接用 `G_t`：
- `G_t = 20` -> 梯度放大 20 倍
- `G_t = -5` -> 梯度缩小（甚至反向）

如果改用 `A(s,a) = G_t - V(s)`：
- `A = 10` -> 正向调整，幅度适中
- `A = -15` -> 负向调整，幅度适中

**方差降低的关键：**

从统计角度看，减去均值（或均值的估计）是最小化方差的经典技巧。`V(s)` 本质上就是条件期望：

```math
V(s) = \mathbb{E}_{\pi_\theta}[G_t | s]
```

所以用 `A(s,a) = G_t - V(s)` 代替 `G_t`，相当于把随机变量中心化，方差自然下降。

#### 5.4 可视化：方差对比图

```text
方差的直观对比：

纯 REINFORCE (用 G_t):
|----波动范围很大----|
[ -30, -20, -10, 0, 10, 20, 30 ]
              ↑ 均值可能不是 0

用 Advantage (A = G_t - V(s)):
|--波动范围小一些--|
[ -15, -10, -5, 0, 5, 10, 15 ]
               ↑ 均值为 0，方差更小
```

**结论：**

> **减去 `V(s)` 作为 baseline，不改变梯度方向，但显著降低方差。**

---

### 6. Actor-Critic 的完整推导：从 TD error 到策略更新

现在我们把所有拼图拼起来，看一个完整的 Actor-Critic 算法长什么样。

#### 6.1 核心组件定义

```text
Actor (策略网络):
    π_θ(a|s) -> 输出动作分布

Critic (价值网络):
    V_w(s) -> 估计状态值
```

参数：
- `θ`: Actor 的参数
- `w`: Critic 的参数

#### 6.2 TD Error 的双重角色

还记得 Bellman 方程吗？

```math
V(s) ≈ r + γ V(s')
```

所以我们可以定义一个**TD error**：

```math
δ = r + γ V_w(s') - V_w(s)
```

这个 `δ` 有两层含义：

| 视角 | 含义 | 用途 |
|------|------|------|
| **Critic 视角** | 当前估计的误差 | 更新 `V_w(s)` 使其更接近真实值 |
| **Actor 视角** | Advantage 的近似 | 作为策略更新的权重信号 |

这就是为什么 Actor-Critic 如此优雅：

> **同一个量，同时服务两个模块。**

#### 6.3 完整的算法流程（伪代码）

```python
# 初始化 Actor π_θ 和 Critic V_w
for episode in range(num_episodes):
    s = env.reset()
    
    for t in range(max_steps):
        # 1. Actor 根据当前策略采样动作
        a ~ π_θ(·|s)
        
        # 2. 执行动作，观察奖励和下一个状态
        r, s_next = env.step(a)
        
        # 3. Critic 评估：计算 TD error
        δ = r + γ * V_w(s_next) - V_w(s)
        
        # 4a. Critic 更新：最小化 TD error 的平方
        L_critic = (δ)^2
        w <- w - α_c * ∇_w L_critic
        
        # 4b. Actor 更新：用 δ 作为 advantage 近似
        grad_actor = δ * ∇_θ log π_θ(a|s)
        θ <- θ + α_a * grad_actor
        
        s = s_next
```

**关键观察：**

- Critic 的梯度是 `∇_w (r + γ V_w(s') - V_w(s))^2`，直接最小化 TD error
- Actor 的梯度是 `δ ∇_θ log π_θ(a|s)`，用 δ 作为 advantage 近似
- **两个更新同时进行**，互相配合

#### 6.4 算法对比：REINFORCE vs Actor-Critic

```text
┌─────────────────────┬──────────────────┬─────────────────────┐
│     特性            │   REINFORCE      │    Actor-Critic     │
├─────────────────────┼──────────────────┼─────────────────────┤
│ 更新信号            │ G_t (累计回报)    │ δ (TD error)        │
│ 方差                │ 高               │ 较低                │
│ 偏差                │ 无 (Monte Carlo)  │ 有 (Bootstrap)      │
│ 样本效率            │ 低               │ 较高                │
│ 是否需要episode结束│ 是               │ 否 (在线更新)       │
└─────────────────────┴──────────────────┴─────────────────────┘
```

**可视化对比：**

```text
REINFORCE 的学习节奏：

Episode 1: [全部走完] -> 拿到 G_t -> 一次性更新
          |--------------------------------|
          
Episode 2: [全部走完] -> 拿到 G_t -> 一次性更新
          |--------------------------------|

Actor-Critic 的学习节奏：

Step 1:   s0 -> a0 -> r0, s1 -> δ0 -> 立即更新
Step 2:   s1 -> a1 -> r1, s2 -> δ1 -> 立即更新
Step 3:   s2 -> a2 -> r2, s3 -> δ2 -> 立即更新
          ^^^ 每一步都能学，不用等 episode 结束 ^^^
```

**结论：**

> **Actor-Critic 用 TD bootstrap 换取了在线学习和更高的样本效率。**

---

### 7. Verification：怎么确认 Actor-Critic 真解决了前一章的问题？

#### 7.1 它有没有保留"直接学策略"的优点？

**有。**

Actor 仍然在直接优化：

```math
π_θ(a|s)
```

所以：
- ✅ 连续动作仍然友好
- ✅ 策略分布仍然可以直接建模
- ✅ 可以表达多模态动作（比如同时向左或向右）

#### 7.2 它有没有让更新信号更局部、更稳定？

**有。**

对比：

| 方法 | 使用信号 | 方差水平 |
|------|----------|---------|
| REINFORCE | G_t (整条轨迹) | 高 |
| Actor-Critic | δ (单步 TD error) | 较低 |

**直观理解：**

```text
REINFORCE: "这局最终得了 100 分，所以你前面所有动作都乘以 1.2"
Actor-Critic: "你刚才这一步比平均水平好 5 分，稍微加强一下"
```

后者的反馈更精细、更稳定。

#### 7.3 它还能不能区分"比平均好"还是"比平均差"？

**能。**

这正是 `δ` 或 `A(s,a)` 的意义：
- `δ > 0` -> 这个动作比预期好，应该更常做
- `δ < 0` -> 这个动作比预期差，应该少做

所以 Actor 不再是被"整局总分"粗暴驱动，而是在接受更细粒度的相对反馈。

#### 7.4 它有没有新的代价？

**有。**

你现在不是只训练一个策略网络，而是要让：
- Actor (θ)
- Critic (w)

同时学习、互相配合。

**潜在问题：**

1. **Critic 评得不准会误导 Actor**  
   - 如果 `V_w(s)` 估计偏差大，`δ` 就会有偏
   - Actor 可能被带偏到错误方向

2. **两个网络的训练节奏要平衡**  
   - Critic 更新太快：Actor 还没适应新策略
   - Critic 更新太慢：反馈信号不准

3. **实现复杂度上升**  
   - 需要维护两套参数
   - 调参空间翻倍（学习率、batch size 等）

**验证结论可以压成一句：**

> **Actor-Critic 用额外的评价模块和实现复杂度，换来了更稳定的策略学习信号和更高的样本效率。**

---

### 8. Example：双足机器人学走路 🔥

这个例子特别适合说明为什么 Actor-Critic 比纯 Policy Gradient 更合理。

#### 8.1 任务长什么样？

**状态空间 (s)**:
- 身体姿态角（倾斜角度）
- 关节角度（髋、膝、踝）
- 关节速度（角速度）
- 足底接触信息（是否着地、受力大小）

**动作空间 (a)**:
- 每个关节的连续扭矩输出 `τ ∈ ℝ^d`
- 假设 10 个自由度，就是输出一个 10 维向量

**奖励函数 (r)**:
```python
def compute_reward(state, action):
    # 往前走给正奖励（基于位移）
    r_forward = velocity_x * 10.0
    
    # 摔倒惩罚（倾斜角太大）
    if abs(tilt_angle) > 30°:
        r_fall = -100.0
    else:
        r_fall = 0.0
    
    # 动作平滑性惩罚（避免抖动）
    r_smooth = -0.01 * ||action||^2
    
    return r_forward + r_fall + r_smooth
```

#### 8.2 为什么纯 Policy Gradient 会很难受？

**early stage 几乎一直在摔。**

如果用 REINFORCE：

```text
Episode: 机器人刚站起来 -> 0.5 秒后摔倒
Reward: -100 (最终)
Update: 所有动作都被标记为"糟糕"
结果: 可能连站都学不会
```

**问题在于：**
- 前 10 帧可能都在正常站立
- 但最后 1 帧因为某个小失误摔倒
- **整条轨迹的负奖励让前面所有好动作也被否定**

这就是 credit assignment 失败的经典场景。

#### 8.3 Actor-Critic 在这里怎么工作？

让我们看看每一步发生了什么：

```text
时间线:
t=0:   机器人直立，Actor 输出初始扭矩
       Critic: V(s_0) ≈ 50 (还能撑一会儿)
       
t=1:   执行动作，身体轻微前倾
       Reward: r_1 = +2 (往前走了一点)
       Critic: V(s_1) ≈ 48 (稍微有点危险)
       TD error: δ_1 = 2 + γ*48 - 50 ≈ 0.5
       Actor: "哦，这一步比预期好一点点" -> 微调策略

t=2:   身体继续前倾
       Reward: r_2 = +3 (又往前走了一点)
       Critic: V(s_2) ≈ 45 (更危险了)
       TD error: δ_2 = 3 + γ*45 - 48 ≈ -1.5
       Actor: "哎呀，这一步比预期差" -> 调整策略避免继续前倾

...

t=50:  终于控制不住，摔倒
       Reward: r_50 = -100
       Critic: V(s_51) ≈ 0 (game over)
       TD error: δ_50 = -100 + γ*0 - 20 ≈ -120
       Actor: "完蛋了，最后一步超级差" -> 大幅调整

但关键在于：前面 t=1,2,...49 的更新已经让它学到了一些东西！
```

**对比 REINFORCE vs Actor-Critic:**

```text
REINFORCE (整局视角):
Episode 最终 reward: -100
所有动作的权重都是 -100 -> "全部都不行"
学习到的：可能连站立都忘了

Actor-Critic (逐帧视角):
t=1:   δ = +0.5  -> "这步还行，保留"
t=2:   δ = -1.5  -> "这步有点糟，改一下"
...
t=49:  δ = -3.0  -> "快撑不住了，赶紧调整"
t=50:  δ = -120  -> "完了完了，重新来过"

即使最后摔了，中间那些稍微正确的动作已经被记住了！
```

#### 8.4 训练曲线对比（示意）

```text
平均 Episode Reward vs Training Steps

REINFORCE:
    |                                   
-100|                 ▓▓▓▓
    |              ▓▓▓▓▓▓▓▓
    |           ▓▓▓▓     ▓▓▓▓▓
    |        ▓▓▓▓              ▓▓▓
    |     ▓▓                         ▓▓▓▓  (波动极大，收敛慢)
    +---------------------------------------->
      0   1k   2k   3k   4k   5k   Steps

Actor-Critic:
    |                                   
-100|              ▓▓▓▓                 
    |           ▓▓▓▓     ▓▓             
    |        ▓▓                 ▓▓      
    |     ▓▓                        ▓▓  
    |  ▓▓                               ▓▓▓▓ (更平滑，收敛更快)
    +---------------------------------------->
      0   1k   2k   3k   4k   5k   Steps
```

**结论：**

> **Actor-Critic 通过逐帧的局部反馈，让机器人能够在"摔倒"的大背景下仍然学到有用的子技能，最终组合成完整的走路能力。**

---

### 9. Actor-Critic 家族：从基础到现代变体

#### 9.1 A2C (Async Advantage Actor-Critic)

**核心思想：**
- 多个 Actor-Critic 实例并行运行
- 每个 worker 收集自己的 trajectory
- 异步更新全局网络参数

```text
Worker 1:  s0->a0->r0,s1 -> δ1 -> 积累梯度
Worker 2:  s0->a0->r0,s1 -> δ2 -> 积累梯度
Worker 3:  s0->a0->r0,s1 -> δ3 -> 积累梯度
   ↓
  Sync: 平均梯度，更新全局 θ, w
```

**优点：**
- 采样效率大幅提升
- 适合分布式训练

**缺点：**
- 异步更新可能引入噪声
- 实现复杂度高

#### 9.2 PPO (Proximal Policy Optimization)

PPO 在 Actor-Critic 的基础上加了两个关键改进：

1. **Clipped surrogate objective**: 限制策略更新的幅度，避免一步更新太大
2. **Generalized Advantage Estimation (GAE)**: 更准确地估计 advantage（结合多步 bootstrap）

**为什么 PPO 成为主流？**
- 稳定、易用、性能好
- 对超参数不那么敏感
- 实现相对简单

#### 9.3 Actor-Critic vs DQN：核心差异对比

```text
┌─────────────────────┬──────────────────┬─────────────────────┐
│     维度            │   Actor-Critic   │      DQN            │
├─────────────────────┼──────────────────┼─────────────────────┤
│ 动作空间            │ 连续友好         │ 离散                │
│ 价值函数            │ V(s) 或 Q(s,a)    │ Q(s,a)              │
│ 策略输出            │ 直接输出分布     │ argmax Q(s,a)       │
│ 探索方式            │ 策略本身的随机性 │ ε-greep / 噪声      │
│ 样本效率            │ 较高             │ 中等                │
└─────────────────────┴──────────────────┴─────────────────────┘

视觉化对比:

Actor-Critic (连续控制):
s -> [Actor] -> π_θ(a|s) -> a ∈ ℝ^d (连续向量)
     [Critic] -> V_w(s)

DQN (离散动作):
s -> [Q 网络] -> Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)
              -> argmax -> a = aᵢ (选择最佳离散动作)
```

---

### 10. 本章速记卡片 🔥

#### 核心直觉

> Policy Gradient 方向是对的，但信号太吵。加一个 Critic 给 Actor 更稳定的反馈，关键桥梁是 `Advantage`。

#### 三个公式就够了

```math
A(s,a) = Q(s,a) - V(s) \quad \text{（比平均好多少）}
```

```math
δ_t = r_t + γ V_w(s_{t+1}) - V_w(s_t) \quad \text{（TD error 近似 advantage）}
```

```math
∇_θ J(θ) ≈ 𝔼[A_t ∇_θ log π_θ(a_t|s_t)] \quad \text{（策略更新方向）}
```

#### 五句话记住本章

1. Actor 负责行动，Critic 负责评价  
2. Advantage = "这一步比平均好多少"  
3. TD error 是 advantage 的便宜近似  
4. 信号更局部了，方差更小  
5. 代价是两个模块会互相影响  

#### 关键对比表格

```text
┌───────────────┬────────────────┬──────────────────┐
│    特性       │   REINFORCE    │   Actor-Critic   │
├───────────────┼────────────────┼──────────────────┤
│ 更新信号      │ G_t (累计)     │ δ_t (单步)       │
│ 方差          │ 高             │ 较低             │
│ 偏差          │ 无             │ 有               │
│ 样本效率      │ 低             │ 较高             │
│ 在线更新      │ ❌ 需要等 episode | ✅ 每一步都可以    │
└───────────────┴────────────────┴──────────────────┘
```

---

### 11. 下一步：PPO 如何从 Actor-Critic 长出来？

你已经掌握了 Actor-Critic 的核心思想。下一章会看到：

- **为什么需要限制策略更新幅度？**（稳定性问题）
- **Clipped surrogate objective 是怎么推导出来的？**（第一性原理视角）
- **GAE (Generalized Advantage Estimation) 是什么？**（更精确的 advantage 估计）

**思考题：**
1. Actor-Critic 中，如果 Critic 学习得太慢会发生什么？
2. 为什么 PPO 比原始的 AC 更稳定？可能的原因有哪些？
3. 如果用 `Q(s,a)` 而不是 `V(s)` 作为 Critic，会有什么好处和代价？

---

#### 核心直觉

Policy Gradient 方向是对的，但信号太吵。加一个 Critic 给 Actor 更稳定的反馈，关键桥梁是 `Advantage`。

#### 三个公式就够了

```math
A(s,a) = Q(s,a) - V(s) \quad \text{（比平均好多少）}
```

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{（TD error 近似 advantage）}
```

```math
\nabla_\theta J(\theta) \approx \mathbb{E}[A_t \nabla_\theta \log \pi_\theta(a_t|s_t)] \quad \text{（策略更新方向）}
```

#### 五句话记住本章

1. Actor 负责行动，Critic 负责评价  
2. Advantage = "这一步比平均好多少"  
3. TD error 是 advantage 的便宜近似  
4. 信号更局部了，方差更小  
5. 代价是两个模块会互相影响  

---

---

