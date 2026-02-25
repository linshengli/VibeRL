# 从零复现：Vibe Coding Agent 到 RL 后训练 — 完整学习与实施指南

> 基于 gaocegege 博文的复现路径，覆盖 Multi-Agent 架构、SFT、GRPO/NGRPO 全流程

---

## 一、学习资料总览

### 1.1 基础知识（先修）

| 主题 | 资料 | 链接 |
|------|------|------|
| Transformer 架构 | Attention Is All You Need | https://arxiv.org/abs/1706.03762 |
| RL 基础 | Spinning Up in Deep RL (OpenAI) | https://spinningup.openai.com/ |
| RLHF 综述 | Training Language Models to Follow Instructions (InstructGPT) | https://arxiv.org/abs/2203.02155 |
| LLM 微调入门 | Hugging Face PEFT 文档 | https://huggingface.co/docs/peft |
| FSDP 原理 | PyTorch FSDP 官方教程 | https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html |
| Tool-use 与 Function Calling | OpenAI Function Calling 文档 | https://platform.openai.com/docs/guides/function-calling |

### 1.2 核心论文（必读）

| 论文 | 关键内容 | 链接 |
|------|----------|------|
| DeepSeekMath (GRPO) | Group Relative Policy Optimization，去掉 critic 网络，用组内相对奖励 | https://arxiv.org/abs/2402.03300 |
| DeepSeek-R1 | 大规模 RL 训练推理模型的实践 | https://arxiv.org/abs/2501.12948 |
| NGRPO | 解决 GRPO 中同质化组奖励方差低的问题，虚拟满分样本 | https://arxiv.org/abs/2509.18851 |
| 推理与工具使用的梯度冲突 | DART：将推理和工具调用解耦训练 | https://arxiv.org/abs/2602.00994 |
| ReAct | Reasoning + Acting 范式 | https://arxiv.org/abs/2210.03629 |
| DPO | Direct Preference Optimization，简化 RLHF | https://arxiv.org/abs/2305.18290 |

### 1.3 框架与工具

| 工具 | 用途 | 链接 |
|------|------|------|
| **verl** | RL 后训练框架（推荐入门，SFT 用 torchrun，RL 用 Ray） | https://github.com/volcengine/verl |
| **slime** | 后训练框架（Megatron + SGLang，适合大规模） | https://github.com/THUDM/slime |
| **OpenRLHF** | 易用的 RL 框架（PPO, DAPO, REINFORCE++） | https://github.com/OpenRLHF/OpenRLHF |
| **Tinker** | 降低后训练门槛的 API | https://tinker-docs.thinkingmachines.ai/ |
| **MoonPalace** | Agent 调试工具（请求捕获） | https://github.com/MoonshotAI/moonpalace |
| **vLLM** | 高效推理引擎，verl 的 rollout 依赖 | https://github.com/vllm-project/vllm |
| **Weights & Biases** | 实验追踪与可视化 | https://wandb.ai/ |

### 1.4 推荐教程

- Hugging Face 后训练算法全景：https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms
- Philipp Schmid - 2025 DPO + 合成数据对齐：https://www.philschmid.de/rl-with-llms-in-2025-dpo
- verl 官方文档与 Recipes：https://verl.readthedocs.io/
- Tinker Cookbook（后训练方法实现集合）：https://github.com/thinking-machines-lab/tinker-cookbook

---

## 二、技术实现路径

整个复现分为 **5 个阶段**，建议按顺序执行：

### Phase 0：环境准备（1-2 天）

**硬件需求：**
- 最低配置：2× NVIDIA L4 (24GB) 或 1× A100 (80GB)
- 推荐配置：4× L4 或 2× A100（RL 训练需要更多显存）
- CPU 内存：≥ 64GB（用于 FSDP CPU offload）
- 磁盘：≥ 200GB SSD（模型权重 + 训练数据 + checkpoints）

**软件环境：**
```bash
# 基础环境
Python 3.10+
CUDA 12.1+
PyTorch 2.1+

# 核心框架
pip install verl          # 后训练框架
pip install vllm          # 推理引擎
pip install wandb         # 实验追踪
pip install transformers datasets accelerate

# Agent 开发
pip install openai        # OpenAI 兼容 API 客户端
pip install akshare       # A 股数据（如复现股票 agent）
```

**模型选择：**
- 基础模型推荐：`Qwen2.5-7B-Instruct` 或 `DeepSeek-R1-Distill-Qwen-7B`
- 数据生成用大模型：`deepseek-chat`（DeepSeek V3）API
- Hugging Face 下载：https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

---

### Phase 1：构建 Tool-use Agent（2-3 天）

**目标：** 用大模型 API 实现一个能正确使用工具的单 Agent

**步骤：**

1. **定义工具集（Tools）**
   - 以股票分析为例：`search_stock_by_name`、`get_stock_info`、`get_realtime_quote` 等
   - 工具需要同时支持 A 股和港股，制造歧义场景用于后续训练
   - 每个工具写好 JSON Schema 描述

2. **实现 ReAct 循环**
   ```
   用户输入 → LLM 推理 → 工具调用 → 获取结果 → LLM 再推理 → ... → 最终输出
   ```
   - 使用 OpenAI 兼容 API（DeepSeek 或 Qwen）
   - 实现多轮 tool calling 的对话管理

3. **搭建调试工具**
   - 参考 MoonPalace，用 SQLite 记录每次 API 请求/响应
   - 关键字段：timestamp、request_body、response_body、tool_calls、duration
   - 提供按时间/request_id 查询的 Web UI

4. **建立评测基准**
   - 准备 50-100 个测试用例（覆盖 A 股、港股、模糊查询等）
   - 定义成功标准：工具调用正确 + 最终输出合理
   - 记录 `deepseek-chat` 的基线通过率（目标 ~80%）

**关键代码结构：**
```
stock_agent/
├── agent.py          # ReAct 主循环
├── tools/
│   ├── search.py     # 股票搜索
│   ├── info.py       # 基本信息
│   ├── quote.py      # 实时行情
│   └── technical.py  # 技术指标
├── debugger/
│   ├── proxy.py      # HTTP 代理记录请求
│   └── viewer.py     # Web UI 查看记录
├── eval/
│   ├── test_cases.json
│   └── evaluate.py
└── config.py
```

---

### Phase 2：SFT 数据构造与训练（3-5 天）

**目标：** 用大模型生成训练数据，SFT 微调 7B 模型

**2.1 数据生成**

用 `deepseek-chat` 生成 ~1000 条高质量多轮对话数据：

```python
# 数据格式（verl SFT 格式）
{
    "messages": [
        {"role": "system", "content": "你是一个股票分析助手..."},
        {"role": "user", "content": "帮我查一下茅台的行情"},
        {"role": "assistant", "content": "<think>用户问的是茅台，A股代码600519...</think>",
         "tool_calls": [{"function": {"name": "get_realtime_quote", "arguments": "{\"stock_code\": \"600519\"}"}}]},
        {"role": "tool", "content": "{\"price\": 1680.00, ...}"},
        {"role": "assistant", "content": "茅台当前价格..."}
    ]
}
```

**数据生成策略：**
- 覆盖多种场景：纯 A 股、纯港股、A 股/港股混淆、多只股票对比
- 保证推理链质量：每条数据的 `<think>` 部分要有清晰的推理过程
- 对易混淆场景做 **过采样**（如 A 股 vs 港股区分）
- 使用 SMOTE 或手动增强少数类样本

**2.2 SFT 训练**

使用 verl 的 SFT 功能（底层 torchrun，不依赖 Ray）：

```bash
# verl SFT 示例命令
torchrun --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --train_files data/sft_train.parquet \
    --val_files data/sft_val.parquet \
    --output_dir checkpoints/sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --gradient_checkpointing true \
    --fsdp_offload true
```

**关键参数说明：**
- `gradient_checkpointing: true` — 必须开启，否则双卡跑不起来
- `fsdp_offload: true` — 显存不够时 offload 到 CPU
- 学习率建议 1e-5 ~ 5e-5
- Epoch 数 2-3 即可，过多会过拟合

**2.3 SFT 评估**

- 在测试集上跑完整的 Agent 流程
- 对比 SFT 前后的工具调用准确率和最终输出质量
- **注意观察**：推理能力与工具使用能力是否此消彼长（梯度冲突问题）

---

### Phase 3：GRPO / NGRPO 强化学习训练（5-7 天）

**目标：** 用 RL 进一步提升模型在 Agent 场景下的表现

**3.1 奖励函数设计**

这是 RL 训练中最关键的部分：

```python
def compute_reward(trajectory):
    """
    对一条完整的 Agent 交互轨迹打分
    """
    reward = 0.0

    # 1. 工具调用正确性（0-0.4 分）
    for tool_call in trajectory.tool_calls:
        if is_correct_tool(tool_call, trajectory.context):
            reward += 0.1
        if has_correct_params(tool_call):
            reward += 0.1

    # 2. 最终输出质量（0-0.4 分）
    # 可以用 LLM-as-Judge 或规则匹配
    if final_output_contains_key_info(trajectory.output):
        reward += 0.4

    # 3. 效率惩罚（-0.2 ~ 0）
    if trajectory.num_turns > expected_turns:
        reward -= 0.1 * (trajectory.num_turns - expected_turns)

    return clip(reward, 0, 1)
```

**奖励设计建议：**
- 组合使用规则奖励（tool 调用是否正确）和 LLM-as-Judge（输出质量）
- 避免纯 sparse reward（只看最终对错），尽量给中间步骤也打分
- 参考微软的稠密 reward 思路：每个 step 都有 reward signal

**3.2 verl GRPO 训练配置**

```yaml
# verl GRPO 核心配置（约 50+ 超参数，以下为关键项）

# Actor 模型配置
actor_rollout_ref:
  actor:
    model_path: checkpoints/sft  # SFT 后的 checkpoint
    fsdp_config:
      param_offload: false       # Actor 不 offload（需要频繁更新）
      optimizer_offload: true    # 优化器可以 offload
  ref:
    fsdp_config:
      param_offload: true        # Ref model 可以 offload
      optimizer_offload: true

# GRPO 算法参数
algorithm:
  grpo:
    group_size: 8                # 每个 prompt 采样 8 个回答
    kl_coef: 0.01               # KL 散度系数
    clip_ratio: 0.2             # PPO clip 比率
    temperature: 0.7            # 采样温度

# 训练参数
trainer:
  total_epochs: 5
  batch_size: 64
  learning_rate: 5e-7           # RL 阶段学习率要低
  gradient_checkpointing: true
```

**⚠️ 常见陷阱：**
- **不要混淆 actor 和 ref 的 offload 设置**：actor 需要频繁更新，offload 会严重拖慢训练；ref model 不更新，可以 offload
- **reward_std 很低 / frac_reward_zero_std 很高**：说明组内回答同质化，模型不学习 → 需要 NGRPO

**3.3 NGRPO 实现**

当遇到 reward 方差过低时，切换到 NGRPO：

```python
# NGRPO 核心修改：在 GRPO 的 advantage 计算中加入虚拟满分样本
def compute_ngrpo_advantages(rewards, max_reward=1.0):
    """
    在每组 rewards 中添加一个虚拟满分样本，
    强行拉大组内方差，让模型有学习信号
    """
    augmented = torch.cat([rewards, torch.tensor([max_reward])])
    mean = augmented.mean()
    std = augmented.std()

    # 只对真实样本计算 advantage（去掉虚拟样本）
    advantages = (rewards - mean) / (std + 1e-8)
    return advantages
```

参考实现：https://github.com/nangongrui-ngr/NGRPO/blob/ngr/verl/trainer/ppo/core_algos.py#L156

**3.4 Multi-turn Agent RL 的特殊处理**

```
# 环境交互流程
for each prompt in batch:
    trajectory = []
    obs = env.reset(prompt)

    for step in range(max_steps):
        action = actor.generate(obs)           # 模型生成（可能是文本或 tool call）

        if action.is_tool_call:
            tool_result = env.execute_tool(action)
            obs = obs + action + tool_result   # 累加上下文
        else:
            break  # 模型给出最终回答

    reward = compute_reward(trajectory)
```

---

### Phase 4：Multi-Agent 架构扩展（可选，5-7 天）

**目标：** 将训练好的小模型用作主 Agent 的状态机管理器

**架构设计：**

```
┌─────────────────────────────────────────┐
│           Main Agent (7B 小模型)          │
│  - 维护状态机（形式化描述）                   │
│  - 决策规划                                │
│  - 分发任务给 SubAgent                     │
├─────────────────────────────────────────┤
│                    │                     │
│    ┌───────────┐   │   ┌───────────┐    │
│    │ SubAgent A│   │   │ SubAgent B│    │
│    │ (大模型)   │   │   │ (大模型)   │    │
│    │ 执行具体   │   │   │ 执行具体   │    │
│    │ 工具调用   │   │   │ 工具调用   │    │
│    └───────────┘   │   └───────────┘    │
└─────────────────────────────────────────┘
```

**状态机形式化描述示例：**
```
STATE: analyzing_stock
  INPUT: user_query("查询茅台和腾讯的对比")
  TRANSITIONS:
    → dispatch_to(SubAgent_AStock, "茅台", "600519")
    → dispatch_to(SubAgent_HKStock, "腾讯", "00700")
    → NEXT_STATE: waiting_for_results

STATE: waiting_for_results
  INPUT: subagent_result(SubAgent_AStock, result_a)
  INPUT: subagent_result(SubAgent_HKStock, result_b)
  TRANSITIONS:
    IF all_results_received:
      → NEXT_STATE: generating_comparison
    ELSE:
      → STAY
```

**训练目标：** 让 7B 模型学会根据当前状态和 SubAgent 输出，正确更新状态机并做出下一步决策，而不会"人格漂移"去接管 SubAgent 的工作。

---

### Phase 5：评估与迭代（持续）

**评估维度：**

| 维度 | 指标 | 目标 |
|------|------|------|
| 工具调用准确率 | 正确调用工具的比例 | ≥ 85% |
| 最终输出质量 | LLM-as-Judge 评分 | ≥ 4/5 |
| 推理一致性 | 推理链是否自洽 | 无反复确认 |
| 延迟 | 端到端响应时间 | < 10s (7B) |
| A/港股区分 | 混淆场景正确率 | ≥ 90% |

---

## 三、需求文档

### 3.1 项目概述

**项目名称：** 从 Vibe Coding Agent 到 RL 后训练——全流程复现

**项目目标：** 完整复现从 Agent 构建、SFT 微调到 GRPO/NGRPO 强化学习训练的全流程，最终得到一个在特定任务（股票分析 Agent）上表现接近大模型的 7B 小模型。

**成功标准：**
- SFT 后 7B 模型工具调用准确率从 ~60% 提升至 ~75%
- GRPO/NGRPO 后进一步提升至 ~85%，接近 deepseek-chat 基线
- 完成从数据生成、训练到部署的端到端流程

### 3.2 功能需求

#### FR-1: 股票分析 Agent
- **FR-1.1**: 支持通过股票代码或名称查询 A 股和港股
- **FR-1.2**: 返回基本信息、实时行情、技术指标
- **FR-1.3**: 支持多轮对话，正确维护上下文
- **FR-1.4**: 工具调用遵循 OpenAI Function Calling 协议

#### FR-2: 训练数据生成
- **FR-2.1**: 自动化生成 1000+ 条 SFT 训练数据
- **FR-2.2**: 数据覆盖 A 股、港股、混淆场景
- **FR-2.3**: 包含推理链（CoT）的多轮对话格式
- **FR-2.4**: 支持 SMOTE 或手动过采样策略

#### FR-3: SFT 训练
- **FR-3.1**: 使用 verl 框架进行 SFT
- **FR-3.2**: 支持 FSDP 分片 + CPU offload
- **FR-3.3**: 支持 gradient checkpointing
- **FR-3.4**: 训练过程可视化（wandb）

#### FR-4: RL 训练
- **FR-4.1**: 实现 GRPO 训练流程
- **FR-4.2**: 实现 NGRPO（虚拟满分样本）以解决低方差问题
- **FR-4.3**: 支持自定义奖励函数
- **FR-4.4**: 支持 multi-turn agent 环境交互

#### FR-5: 评估系统
- **FR-5.1**: 自动化评测 pipeline
- **FR-5.2**: 多维度评分：工具准确率、输出质量、推理一致性
- **FR-5.3**: A/B 对比：SFT vs RL vs 基线大模型

#### FR-6: 调试工具
- **FR-6.1**: HTTP 代理记录所有 LLM API 请求/响应
- **FR-6.2**: Web UI 查看请求详情
- **FR-6.3**: 支持按时间、request_id 查询

### 3.3 非功能需求

#### NFR-1: 硬件资源
- 训练：4× L4 (24GB) 或 2× A100 (80GB)
- 推理：1× L4 或 T4 即可
- 内存：≥ 64GB
- 存储：≥ 200GB SSD

#### NFR-2: 训练效率
- SFT：1000 条数据，3 epochs，< 2 小时
- GRPO：< 24 小时完成 5 epochs
- 支持断点续训

#### NFR-3: 可复现性
- 所有实验参数记录到 wandb
- 训练脚本版本控制（Git）
- 数据集固定 seed 生成

### 3.4 里程碑与时间表

| 里程碑 | 交付物 | 预计耗时 |
|--------|--------|----------|
| M0: 环境搭建 | GPU 服务器就绪、依赖安装完成 | 1-2 天 |
| M1: Agent 原型 | 可运行的股票分析 Agent + 调试工具 | 2-3 天 |
| M2: 数据集 | 1000+ 条 SFT 数据 + 测试集 | 2-3 天 |
| M3: SFT 完成 | 微调后的 7B 模型 + 评估报告 | 3-5 天 |
| M4: RL 完成 | GRPO/NGRPO 训练后模型 + 评估报告 | 5-7 天 |
| M5: 集成测试 | 端到端 Agent 运行 + 对比分析文档 | 2-3 天 |
| **总计** | | **15-23 天** |

### 3.5 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 显存不足 | 训练无法启动 | 开启 gradient checkpointing + FSDP offload |
| Reward 方差过低 | 模型不学习 | 切换 NGRPO / 调整奖励函数 |
| 推理与工具使用梯度冲突 | SFT 后能力此消彼长 | 参考 DART 论文，解耦训练（LoRA） |
| 数据质量差 | 训练效果不佳 | 增加数据验证步骤，人工抽检 |
| verl 版本兼容 | 框架 API 变化 | 锁定 verl 版本，参考官方 recipes |

---

## 四、快速开始 Checklist

```
□ 1. 租 GPU 服务器（推荐 AutoDL / Lambda / vast.ai）
□ 2. 安装 verl + vLLM + PyTorch 环境
□ 3. 下载 Qwen2.5-7B-Instruct 模型权重
□ 4. 获取 DeepSeek API key（用于数据生成）
□ 5. 实现股票分析 Agent 并验证基线
□ 6. 生成 SFT 训练数据
□ 7. 运行 SFT 训练
□ 8. 评估 SFT 效果
□ 9. 设计奖励函数
□ 10. 运行 GRPO 训练
□ 11. 如遇低方差问题，切换 NGRPO
□ 12. 最终评估与对比分析
```

---

*文档生成日期：2026-02-26*
*基于 gaocegege 博文复现路径整理*
