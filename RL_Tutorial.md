# 强化学习教程 (RL Tutorial)

> 本文档为教程索引，各章节内容已拆分到 `chapters/` 目录下。

## 全局导航图

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

## 章节目录

| 章节 | 标题 | 文件 |
|------|------|------|
| 第1章 | RL 是什么？ | [chapter_01_RL是什么？.md](chapters/chapter_01_RL是什么？.md) |
| 第2章 | Policy（策略）π | [chapter_02_Policy（策略）π.md](chapters/chapter_02_Policy（策略）π.md) |
| 第3章 | Value Function 与 Bellman 方程 | [chapter_03_Value_Function与Bellman方程.md](chapters/chapter_03_Value_Function与Bellman方程.md) |
| 第4章 | Q-Learning（让 Agent 开始真正学习） | [chapter_04_Q-Learning.md](chapters/chapter_04_Q-Learning.md) |
| 第5章 | Deep Q-Network (DQN) - 用神经网络替代 Q 表 | [chapter_05_DQN.md](chapters/chapter_05_DQN.md) |
| 第6章 | Policy Gradient - 直接优化策略 | [chapter_06_Policy_Gradient.md](chapters/chapter_06_Policy_Gradient.md) |
| 第7章 | Actor-Critic - 结合 Policy 和 Value | [chapter_07_Actor-Critic.md](chapters/chapter_07_Actor-Critic.md) |
| 第8章 | PPO - 给策略更新套上"安全护栏" | [chapter_08_PPO.md](chapters/chapter_08_PPO.md) |
| 第9章 | LLM 分支入口 - 为什么 GRPO 应该放在 PPO 之后？ | [chapter_09_LLM分支入口.md](chapters/chapter_09_LLM分支入口.md) |
| 第10章 | 从经典 RL 到 LLM 对齐 - 为什么语言模型也会走到 RLHF？ | [chapter_10_从经典RL到LLM对齐.md](chapters/chapter_10_从经典RL到LLM对齐.md) |
| 第11章 | PPO for LLMs - 当动作变成 token，PPO 还在优化什么？ | [chapter_11_PPO_for_LLMs.md](chapters/chapter_11_PPO_for_LLMs.md) |
| 第12章 | GRPO - 在 LLM 场景里，为什么"组内相对比较"会自然替代一部分 critic 角色？ | [chapter_12_GRPO.md](chapters/chapter_12_GRPO.md) |
| 第13章 | DDPG - 为什么连续动作会逼出"确定性 Actor-Critic"？ | [chapter_13_DDPG.md](chapters/chapter_13_DDPG.md) |
| 第14章 | TD3 - 为什么 DDPG 会被"双 Critic + 延迟更新"继续修正？ | [chapter_14_TD3.md](chapters/chapter_14_TD3.md) |
| 第15章 | SAC - 为什么最大熵原则会逼出"既学回报，也保留随机性"的算法？ | [chapter_15_SAC.md](chapters/chapter_15_SAC.md) |
| 第16章 | Outcome Reward - 为什么"只看最终答案对不对"既强大又不够？ | [chapter_16_Outcome_Reward.md](chapters/chapter_16_Outcome_Reward.md) |
| 第17章 | Process Reward - 为什么只奖最终答案还不够？ | [chapter_17_Process_Reward.md](chapters/chapter_17_Process_Reward.md) |
| 第18章 | Verifiable Reward - 为什么 reasoning 训练越来越依赖"可自动检查"的中间与最终信号？ | [chapter_18_Verifiable_Reward.md](chapters/chapter_18_Verifiable_Reward.md) |
| 第19章 | 总复习总览 - 把整本 RL 教程压成一张因果地图 | [chapter_19_总复习总览.md](chapters/chapter_19_总复习总览.md) |

---

## 学习路径建议

### 经典 RL 路径
1. 第1-4章 - 基础概念 → Q-Learning
2. 第5章 - DQN (神经网络版 Q-Learning)
3. 第6-8章 - Policy Gradient → Actor-Critic → PPO

### 连续动作控制路径
- 第13-15章 - DDPG → TD3 → SAC

### LLM / Reasoning 路径
- 第9-12章 - LLM 后训练 → RLHF → GRPO
- 第16-18章 - Outcome → Process → Verifiable Reward

### 总复习
- 第19章 - 整本教程的知识地图

---

*本教程采用"问题驱动式第一性理解"方式编写，每个概念都从"为什么要解决什么问题"出发，而非直接堆砌公式。*
