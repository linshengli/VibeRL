# 功能规格说明：Vibe Coding Agent RL 后训练全流程复现

**功能分支**: `001-vibe-coding-agent-rl`
**创建日期**: 2026-02-26
**状态**: 草稿
**输入来源**: 《从零复现 Vibe Coding Agent 到 RL 后训练 — 完整指南》

---

## 用户场景与测试 *(必填)*

### 用户故事 1 — 股票分析 Agent 原型（优先级：P1）

作为一名 ML 工程师，我希望构建一个能调用真实股票工具的 ReAct Agent，
使其可以正确区分 A 股与港股、执行多步工具调用，并以此作为后续 SFT/RL 训练的环境基础。

**优先级理由**：这是整个训练流水线的入口；没有可运行的 Agent 环境，
SFT 数据无法生成，RL 环境无法搭建。

**独立测试方式**：可单独启动 Agent，输入「查一下茅台和腾讯的行情」，
验证 Agent 正确识别 A 股（600519）与港股（00700），
并依次调用对应工具获取结果，无需任何训练步骤介入。

**验收场景**:

1. **Given** 股票名称模糊（如「平安」同时存在 A 股和港股），
   **When** Agent 调用 `search_stock_by_name`，
   **Then** 返回多个候选，Agent 进一步确认后选择正确市场。
2. **Given** 用户输入「帮我查一下茅台最近的技术指标」，
   **When** Agent 执行完整 ReAct 循环，
   **Then** 工具调用顺序合理（先搜索，再查行情，再查技术指标），
   最终输出包含关键技术指标数值。
3. **Given** API 调用失败，
   **When** Agent 收到工具错误，
   **Then** Agent 进行合理重试或告知用户，不崩溃。

---

### 用户故事 2 — SFT 数据生成与训练（优先级：P2）

作为一名 ML 工程师，我希望用大模型（deepseek-chat）自动生成高质量
多轮对话训练数据，并用 verl 完成 SFT 微调，使 7B 模型的工具调用准确率
从约 60% 提升至约 75%。

**优先级理由**：SFT 是 RL 训练的基础，SFT 后的 checkpoint 是 GRPO 的起点。

**独立测试方式**：可单独运行数据生成脚本，生成 100 条样本，
验证格式符合 verl parquet 规范；
可单独运行 SFT 训练 1 epoch，验证 loss 下降、无 OOM。

**验收场景**:

1. **Given** 1000 条已验证的 SFT 数据，
   **When** 运行 verl SFT 训练 3 epochs，
   **Then** 验证集 loss 持续下降，无 NaN，无 OOM。
2. **Given** SFT 后的模型，
   **When** 在 50 条测试集上运行完整 Agent 评估，
   **Then** 工具调用准确率 ≥ 75%（相比 SFT 前的 ~60%）。
3. **Given** 生成的数据中有少数易混淆场景（A/港股区分），
   **When** 评估 A/港股区分能力，
   **Then** 混淆场景准确率 ≥ 80%。

---

### 用户故事 3 — GRPO/NGRPO 强化学习训练（优先级：P3）

作为一名 ML 工程师，我希望在 SFT 模型基础上运行 GRPO（或 NGRPO）RL 训练，
进一步将工具调用准确率提升至约 85%，接近大模型基线。

**优先级理由**：RL 训练是项目的核心目标，但依赖 SFT 完成。

**独立测试方式**：可单独运行 GRPO 1 个 epoch（small batch），
验证 reward 均值上升、`reward_std` 不为零（若全零则切换 NGRPO）。

**验收场景**:

1. **Given** SFT checkpoint 和已实现的奖励函数，
   **When** 运行 GRPO 训练 5 epochs，
   **Then** reward 均值呈上升趋势，工具调用准确率在测试集上 ≥ 85%。
2. **Given** `frac_reward_zero_std > 0.5`（reward 方差过低），
   **When** 切换至 NGRPO（虚拟满分样本），
   **Then** `reward_std` 恢复正常，模型继续学习。
3. **Given** 训练完成，
   **When** 运行 A/B 对比评估（SFT vs RL vs deepseek-chat 基线），
   **Then** RL 模型在工具准确率、输出质量上均优于 SFT 模型。

---

### 用户故事 4 — 实验调试与可观测（优先级：P2）

作为一名 ML 工程师，我希望有调试工具记录所有 LLM API 请求/响应，
并通过 wandb 追踪训练指标，以便快速定位问题。

**优先级理由**：贯穿整个开发周期，影响所有用户故事的调试效率。

**独立测试方式**：启动调试代理，发送一条 API 请求，
验证请求体和响应体被完整记录到 SQLite，可通过 Web UI 查看。

**验收场景**:

1. **Given** 调试代理已启动，
   **When** Agent 通过代理调用 LLM API，
   **Then** 请求和响应被记录，包含 timestamp、request_id、tool_calls 字段。
2. **Given** wandb 项目已配置，
   **When** SFT 或 RL 训练运行，
   **Then** loss、reward、reward_std 等关键指标实时上传 wandb 可视化。

---

### 边界情况

- 股票数据 API（akshare）限流或返回空数据时，Agent 如何处理？
- RL 训练中途中断（OOM 或异常），是否支持断点续训？
- 数据生成 API（deepseek-chat）超时，批量生成如何保障容错？
- `reward_std = 0` 时，GRPO 不学习，自动切换 NGRPO 的触发逻辑是什么？

---

## 功能需求 *(必填)*

### 功能性需求

- **FR-001**: 系统 MUST 实现 ReAct Agent 循环，支持最大 10 步工具调用
- **FR-002**: 工具集 MUST 包含 `search_stock_by_name`、`get_stock_info`、
  `get_realtime_quote`、`get_technical_indicators`，覆盖 A 股和港股
- **FR-003**: 工具 MUST 以 OpenAI Function Calling JSON Schema 格式定义
- **FR-004**: 系统 MUST 支持 HTTP 代理调试器，记录所有 LLM 请求到 SQLite
- **FR-005**: 数据生成脚本 MUST 输出 verl 兼容的 parquet 格式
- **FR-006**: 系统 MUST 支持通过 verl `fsdp_sft_trainer` 完成 SFT 训练
- **FR-007**: 奖励函数 MUST 包含工具调用正确性、最终输出质量、效率惩罚三个维度
- **FR-008**: 系统 MUST 实现 NGRPO 虚拟满分样本逻辑，可在 GRPO 基础上切换
- **FR-009**: 评估系统 MUST 支持自动化运行测试用例并输出准确率报告
- **FR-010**: 系统 MUST 支持 wandb 实验追踪，所有训练超参可配置化（YAML）

### 核心实体

- **StockTool（股票工具）**: 工具名称、JSON Schema 描述、A/港股支持标记
- **AgentTrajectory（Agent 轨迹）**: prompt、tool_calls 序列、最终输出、奖励分值
- **SFTSample（SFT 样本）**: messages 列表（含 CoT 推理链）、market_type 标签
- **RewardSignal（奖励信号）**: trajectory_id、tool_correctness、output_quality、
  efficiency_penalty、final_reward
- **TrainingConfig（训练配置）**: 模型路径、超参数、FSDP 配置、wandb 项目名

---

## 成功标准 *(必填)*

### 可量化结果

- **SC-001**: SFT 后 7B 模型工具调用准确率从基线 ~60% 提升至 ≥ 75%
- **SC-002**: GRPO/NGRPO 训练后工具调用准确率进一步提升至 ≥ 85%
- **SC-003**: A/港股混淆场景的正确率 ≥ 90%
- **SC-004**: SFT 训练（1000 条数据，3 epochs）在 4× L4 上 < 2 小时完成
- **SC-005**: GRPO 训练（5 epochs）在 4× L4 上 < 24 小时完成
- **SC-006**: 所有训练脚本支持断点续训，中途中断不丢失进度
- **SC-007**: 调试工具成功记录 ≥ 99% 的 API 请求（无丢失）
- **SC-008**: 单元测试覆盖率 ≥ 80%（核心模块：工具调用解析、奖励计算、数据格式化）
