# AGENTS.md

本文件为 AI 编程助手提供项目背景、构建指令和工作约定。

## 项目概述

这是一个**文档优先**的强化学习（Reinforcement Learning）教程项目，采用"问题驱动式第一性理解"方式编写。教程使用中文撰写，包含 19 个章节，涵盖从经典 RL（Q-Learning、DQN、PPO）到 LLM 对齐（RLHF、GRPO）的完整学习路径。

**项目特点：**
- 教程文档使用 Markdown 格式，章节文件位于 `chapters/` 目录
- C++ 示例代码用于演示核心算法（Bellman 方程、Q-Learning）
- 通过 `merge_tutorial.py` 将分散的章节合并为完整文档
- 无正式测试套件，通过运行示例程序和文档生成验证正确性

**学习路径导航图：**
```text
延迟奖励/长远目标 → Value/Bellman/TD → Q-Learning/DQN
                                    │
            ┌───────────────────────┴───────────────────────┐
            ↓                                               ↓
    直接优化策略需求                              连续动作 + 样本效率需求
            ↓                                               ↓
   Policy Gradient                                  DDPG → TD3 → SAC
            ↓
     Actor-Critic
            ↓
           PPO
            │
    ┌───────┴───────┐
    ↓               ↓
 LLM 后训练      经典控制继续
    ↓
 RLHF / PPO for LLMs
    ↓
   GRPO
    ↓
Outcome → Process → Verifiable Reward
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 文档格式 | Markdown |
| 文档构建 | Python 3 (纯标准库，仅 `pathlib` 和 `re`) |
| C++ 构建 | CMake (最低版本 3.14.0) |
| C++ 标准 | C++11/C++14 (使用 `std::shared_ptr`, `auto`, range-based for) |
| 线程库 | pthread |
| Python 依赖 | gymnasium==1.0.0, numpy, matplotlib (仅用于教程代码示例) |

## 项目结构

```
rl_tutorial/
├── RL_Tutorial.md              # 教程索引（含导航图、章节目录、学习路径）
├── RL_Tutorial_Full.md         # 【生成文件】合并后的完整教程
├── merge_tutorial.py           # 文档合并脚本
├── requirements.txt            # Python 依赖
├── chapters/                   # 章节源文件（19 个 Markdown 文件）
│   ├── chapter_01_RL是什么？.md
│   ├── chapter_02_Policy（策略）π.md
│   ├── chapter_03_Value_Function与Bellman方程.md
│   ├── chapter_04_Q-Learning.md
│   ├── ...（共 19 章）
│   └── chapter_19_总复习总览.md
├── 3.0_bellman_cpp_cheese_example/   # 第 3 章配套 C++ 示例：Bellman 方程
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── dag.h                   # DAG 图结构定义（BFS/DFS/路径查找）
│   └── bellman_update.cc       # Bellman 更新与策略提取实现
├── 4.0_q_learning_cpp/              # 第 4 章配套 C++ 示例：Q-Learning
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── q_learning.h            # Q 表定义、动作空间、单元格类型
│   ├── q_learning.cc           # Q 表初始化与更新实现
│   └── main.cc                 # 单元测试与主训练循环
└── bellman_cpp_cheese_example/      # 【旧目录，疑似废弃】
```

## 构建命令

### 文档构建

```bash
# 安装 Python 依赖（如需运行教程中的 Python 示例）
pip install -r requirements.txt

# 生成合并后的完整教程文档
python merge_tutorial.py
```

执行后将生成 `RL_Tutorial_Full.md`，包含所有章节内容。

### C++ 示例构建

```bash
# 构建 Bellman 方程示例（第 3 章）
cd 3.0_bellman_cpp_cheese_example && ./build.sh

# 运行 Bellman 示例
./3.0_bellman_cpp_cheese_example/build/bellman_update_cheese_example

# 构建 Q-Learning 示例（第 4 章）
cd 4.0_q_learning_cpp && ./build.sh

# 运行 Q-Learning 示例
./4.0_q_learning_cpp/build/qlearning
```

**注意：** `build.sh` 脚本会**删除并重新创建**本地 `build/` 目录，然后调用 CMake 编译。

## 代码风格约定

### Markdown 文档

- **文件编码：** UTF-8（章节文件名包含中文）
- **章节编号格式：** `chapter_XX_中文标题.md`
- **章节表格格式：** `RL_Tutorial.md` 中的章节表格必须保持以下格式，否则 `merge_tutorial.py` 无法正确解析：
  ```markdown
  | 章节 | 标题 | 文件 |
  |------|------|------|
  | 第X章 | 章节标题 | [文件名](路径) |
  ```
- **重复内容：** 每章开头包含相同的"全局导航图"ASCII 图表，便于读者定位

### C++ 代码

- **日志宏：** 使用统一的日志宏定义：
  ```cpp
  #define LOG_INFO std::cout << __FILE__ << ":" << __LINE__ << ":"
  #define LOG_ERROR std::cerr << __FILE__ << ":" << __LINE__ << ":"
  ```
- **命名规范：**
  - 类名：`PascalCase`（如 `BellmanDAG`, `QLearning`）
  - 成员变量：`snake_case_` 后缀下划线（如 `gamma_`, `q_table_`）
  - 常量：`kPascalCase`（如 `kActionSpace`, `kRows`）
  - 函数：`PascalCase`（如 `InitializeTable`, `Optimize`）
- **智能指针：** 使用 `std::shared_ptr` 管理图节点和 Q 表
- **随机数：** 使用 `<random>` 库（`std::mt19937`, `std::uniform_int_distribution`）

### Python 代码

- 仅用于文档合并，保持与现有代码风格一致
- 使用 `pathlib.Path` 处理路径
- 使用正则表达式解析 Markdown 表格

## 验证与测试

**本项目无正式测试套件、无 linter、无 CI 配置。**

验证方式：

1. **文档变更验证：**
   ```bash
   python merge_tutorial.py
   # 检查生成的 RL_Tutorial_Full.md 是否正确包含所有章节
   ```

2. **C++ 代码变更验证：**
   ```bash
   # 重新构建受影响的子项目并运行可执行文件
   cd 3.0_bellman_cpp_cheese_example && ./build.sh && ./build/bellman_update_cheese_example
   cd 4.0_q_learning_cpp && ./build.sh && ./build/qlearning
   ```

## 开发工作流

### 编辑章节内容

1. 直接编辑 `chapters/chapter_XX_标题.md` 文件
2. **不要**直接编辑 `RL_Tutorial_Full.md`（它是生成文件）
3. 如需调整章节顺序或标题，同步更新 `RL_Tutorial.md` 中的表格
4. 运行 `python merge_tutorial.py` 验证合并结果

### 编辑教程索引

- `RL_Tutorial.md` 包含全局导航图、章节目录、学习路径建议
- 修改章节表格时，**必须**保持 `| 第X章 | 标题 | [文件名](路径) |` 格式
- 如需修改表格结构，需同步更新 `merge_tutorial.py` 中的正则表达式解析逻辑

### 编辑 C++ 示例

- 编号目录（`3.0_*`, `4.0_*`）与对应章节概念对齐
- 保持教学代码的可读性，优先清晰度而非性能
- 每个示例应能独立构建和运行
- 修改后必须在对应目录运行 `./build.sh` 验证编译通过

## 重要约束

### 文件路径

- 章节文件名和路径包含中文字符和标点（如 `chapter_01_RL是什么？.md`）
- 编辑索引时保持路径精确匹配

### 生成文件

以下文件为派生输出，不应手动编辑：
- `RL_Tutorial_Full.md`
- `3.0_bellman_cpp_cheese_example/build/`
- `4.0_q_learning_cpp/build/`
- `bellman_cpp_cheese_example/build/`

### 运行目录

- 大多数命令应从仓库根目录执行
- C++ 的 `build.sh` 脚本设计为从子项目目录运行（脚本内部会处理路径）

## 章节与代码对应关系

| 章节 | 标题 | 对应代码示例 |
|------|------|-------------|
| 第 3 章 | Value Function 与 Bellman 方程 | `3.0_bellman_cpp_cheese_example/` |
| 第 4 章 | Q-Learning | `4.0_q_learning_cpp/` |

第 5-19 章目前无对应 C++ 示例代码，仅包含 Markdown 文档。
