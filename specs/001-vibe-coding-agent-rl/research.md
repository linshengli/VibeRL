# Phase 0 研究报告：Vibe Coding Agent RL 后训练

**生成日期**: 2026-02-26
**关联功能**: 001-vibe-coding-agent-rl

---

## 技术选型决策

### 决策 1：RL 框架选型

- **决定**: 使用 **verl**（volcengine/verl）作为主训练框架
- **依据**: verl 同时支持 SFT（torchrun，不依赖 Ray）和 RL（Ray + vLLM），
  文档完善，官方提供 GRPO recipes，适合入门级复现。
  slime 适合大规模 Megatron 场景，OpenRLHF 更适合 PPO/DAPO，
  但 verl 的工具链最适合本项目的 7B 量级和 GRPO 目标。
- **备选方案**:
  - OpenRLHF：接口易用，但 GRPO 支持不如 verl 完整。
  - slime：适合大规模，本项目 7B 量级无需 Megatron。
- **参考文档**:
  - verl GitHub: https://github.com/volcengine/verl
  - verl 官方文档: https://verl.readthedocs.io/

### 决策 2：推理引擎选型

- **决定**: 使用 **vLLM** 作为 rollout 推理引擎
- **依据**: verl 的 rollout 模块默认集成 vLLM，提供 PagedAttention 和
  连续批处理，效率显著高于 HuggingFace generate。
  RL 训练中 rollout 是性能瓶颈，vLLM 可加速 2-5 倍。
- **备选方案**: SGLang（更新但与 verl 集成不如 vLLM 成熟）
- **参考文档**: https://docs.vllm.ai

### 决策 3：基础模型选型

- **决定**: 使用 **Qwen2.5-7B-Instruct** 作为基础模型
- **依据**: Qwen2.5-7B 对中文支持优秀，7B 参数在 4× L4 上可运行，
  Instruct 版本已具备 Function Calling 能力，减少 SFT 冷启动难度。
  DeepSeek-R1-Distill-Qwen-7B 为备选（推理能力更强，但工具调用初始能力弱）。
- **备选方案**: DeepSeek-R1-Distill-Qwen-7B（推理链更好但工具调用弱）
- **参考文档**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

### 决策 4：股票数据 API

- **决定**: 使用 **akshare** 获取 A 股数据，使用 **yfinance** 或
  akshare 的港股接口获取港股数据
- **依据**: akshare 是国内最完整的开源 A 股数据库，免费无需 API Key，
  支持实时行情、历史数据、技术指标。
- **备选方案**: baostock（仅历史数据，不支持实时）
- **参考文档**: https://akshare.akfamily.xyz/

### 决策 5：RL 算法选型

- **决定**: 主要使用 **GRPO**，当 `reward_std` 过低时切换 **NGRPO**
- **依据**: GRPO（DeepSeekMath 提出）去掉 critic 网络，减少显存占用，
  适合本项目的 7B 量级。NGRPO 通过虚拟满分样本解决组内同质化问题，
  作为 GRPO 降级备选策略。
- **参考文档**:
  - GRPO: https://arxiv.org/abs/2402.03300（DeepSeekMath）
  - NGRPO: https://arxiv.org/abs/2509.18851
  - NGRPO 参考实现: https://github.com/nangongrui-ngr/NGRPO/blob/ngr/verl/trainer/ppo/core_algos.py#L156

### 决策 6：数据生成策略

- **决定**: 使用 **deepseek-chat（DeepSeek V3）** API 生成 SFT 数据
- **依据**: deepseek-chat 工具调用能力强，生成的推理链（CoT）质量高，
  API 价格合理（约 $0.14/M tokens），可批量生成 1000+ 条数据。
- **备选方案**: GPT-4o（贵，但质量更稳定）；本地 70B 模型（无 API 成本，但生成速度慢）
- **参考**: DeepSeek API 文档: https://platform.deepseek.com/docs

### 决策 7：实验追踪

- **决定**: 使用 **Weights & Biases (wandb)**
- **依据**: verl 原生支持 wandb 集成，配置简单；wandb 支持自定义指标面板，
  可实时监控 reward、loss、reward_std 等关键指标。
- **参考**: https://wandb.ai/

### 决策 8：调试代理方案

- **决定**: 实现轻量级 HTTP 代理 + SQLite 存储 + Flask Web UI
- **依据**: 参考 MoonPalace 思路，但不依赖外部工具。
  SQLite 无需额外部署，Flask 轻量易用，满足本项目调试需求。
- **参考**: https://github.com/MoonshotAI/moonpalace

---

## 关键算法研究摘要

### GRPO（Group Relative Policy Optimization）

**来源**: DeepSeekMath（arXiv:2402.03300）

**核心思想**: 对每个 prompt 采样 G 个回答，用组内平均奖励作为 baseline，
计算相对优势（advantage），避免引入单独的 critic 网络：

```
A_i = (r_i - mean(r)) / std(r)
```

**优点**: 显存占用比 PPO 低约 40%（无 critic 网络），训练稳定。
**缺点**: 当组内回答同质化时，`std(r) ≈ 0`，advantage ≈ 0，模型停止学习。

### NGRPO（解决低方差问题）

**来源**: arXiv:2509.18851

**核心修改**: 在每组奖励中插入一个虚拟满分样本（max_reward=1.0），
强行拉大组内方差：

```python
augmented = torch.cat([rewards, torch.tensor([max_reward])])
advantages = (rewards - augmented.mean()) / (augmented.std() + 1e-8)
```

**适用场景**: `frac_reward_zero_std > 0.3`（超过 30% 的组方差为零）时切换。

### ReAct 范式

**来源**: arXiv:2210.03629（Yao et al.）

**流程**: Thought → Action → Observation 循环，直到输出 Final Answer。
**参考实现**: LangChain / OpenAI Assistants API 均基于此范式。

### DART（解耦推理与工具调用）

**来源**: arXiv:2602.00994

**背景**: SFT 过程中，推理能力（CoT 质量）与工具调用能力可能存在梯度冲突，
提升工具调用准确率可能损伤推理链质量。
**解决方案**: 使用 LoRA 对推理头和工具调用头分别微调（解耦训练）。
**实施建议**: 若 SFT 后发现推理质量明显下降，引入 DART 的解耦策略。

---

## 硬件与环境确认

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| GPU | 2× L4 (24GB) | 4× L4 或 2× A100 (80GB) |
| CPU 内存 | 64GB | 128GB |
| 磁盘 | 200GB SSD | 500GB SSD |
| Python | 3.10+ | 3.11 |
| CUDA | 12.1+ | 12.4 |
| PyTorch | 2.1+ | 2.3 |

**关键约束**:
- Actor 模型 MUST 不开启 param_offload（频繁更新，offload 会拖慢 3-5 倍）
- Ref 模型可以开启 param_offload（不更新，offload 节省显存）
- gradient_checkpointing MUST 开启（否则双卡 24GB 跑不起来 7B 模型）

---

## 所有 NEEDS CLARIFICATION 已解决

| 问题 | 解决方案 |
|------|----------|
| RL 框架选哪个？ | verl（GRPO 支持完整，文档最全） |
| 基础模型用哪个？ | Qwen2.5-7B-Instruct |
| 港股数据来源？ | akshare 港股接口 |
| 数据生成用哪个 API？ | deepseek-chat（DeepSeek V3） |
| 低方差时如何处理？ | 自动切换 NGRPO |
| 调试工具实现方案？ | 自实现 HTTP 代理 + SQLite + Flask |
