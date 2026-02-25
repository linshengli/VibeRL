# 数据模型：Vibe Coding Agent RL 后训练

**生成日期**: 2026-02-26
**关联功能**: 001-vibe-coding-agent-rl

---

## 核心实体定义

### 1. StockTool（股票工具）

Agent 可调用的工具定义。每个工具对应一个 Python 函数，并携带 OpenAI Function
Calling 格式的 JSON Schema 描述。

```python
@dataclass
class StockTool:
    name: str              # 工具名，如 "search_stock_by_name"
    description: str       # 工具用途描述（中文）
    parameters: dict       # JSON Schema 格式的参数定义
    supports_a_share: bool # 是否支持 A 股
    supports_hk_share: bool # 是否支持港股
    handler: Callable      # 工具的 Python 实现函数
```

**工具清单**:

| 工具名 | 描述 | A 股 | 港股 |
|--------|------|------|------|
| `search_stock_by_name` | 通过名称模糊搜索股票代码 | ✓ | ✓ |
| `get_stock_info` | 获取股票基本信息（公司名、行业、市值） | ✓ | ✓ |
| `get_realtime_quote` | 获取实时行情（价格、涨跌幅、成交量） | ✓ | ✓ |
| `get_technical_indicators` | 获取技术指标（MA、MACD、RSI、KDJ） | ✓ | ✓ |

---

### 2. AgentMessage（Agent 消息）

ReAct 循环中的单条消息，对应 OpenAI Chat API 的消息格式。

```python
@dataclass
class AgentMessage:
    role: str              # "system" / "user" / "assistant" / "tool"
    content: Optional[str] # 消息文本内容
    tool_calls: Optional[List[ToolCall]]  # 工具调用请求（role=assistant 时）
    tool_call_id: Optional[str]           # 工具调用 ID（role=tool 时）
    name: Optional[str]    # 工具名（role=tool 时）
```

---

### 3. ToolCall（工具调用）

Agent 请求执行的单次工具调用。

```python
@dataclass
class ToolCall:
    id: str                # 唯一 ID，如 "call_abc123"
    type: str              # 固定为 "function"
    function: FunctionCall # 函数名称和参数

@dataclass
class FunctionCall:
    name: str              # 工具名，如 "get_realtime_quote"
    arguments: str         # JSON 字符串格式的参数，如 '{"stock_code": "600519"}'
```

---

### 4. AgentTrajectory（Agent 执行轨迹）

一次完整的 Agent 交互过程，用于奖励计算和 RL 训练。

```python
@dataclass
class AgentTrajectory:
    trajectory_id: str         # 轨迹唯一 ID（UUID）
    prompt: str                # 用户原始问题
    messages: List[AgentMessage]  # 完整对话历史
    tool_calls: List[ToolCall]    # 所有工具调用（按时间顺序）
    final_output: str          # 模型最终文本回答
    num_turns: int             # Agent 循环轮数
    success: bool              # 是否成功完成任务
    metadata: dict             # 额外信息（如预期工具序列、正确答案）
```

---

### 5. SFTSample（SFT 训练样本）

用于 SFT 训练的单条数据，输出为 verl 兼容的 parquet 格式。

```python
@dataclass
class SFTSample:
    sample_id: str             # 样本唯一 ID
    messages: List[dict]       # OpenAI 格式的 messages 列表
    market_type: str           # "a_share" / "hk_share" / "mixed"（用于过采样）
    difficulty: str            # "easy" / "medium" / "hard"（hard=混淆场景）
    has_cot: bool              # 是否包含 <think> 推理链
    source_model: str          # 生成模型（如 "deepseek-chat"）
    generated_at: str          # ISO 8601 时间戳
```

**verl parquet 字段映射**:

| parquet 列名 | SFTSample 字段 | 说明 |
|-------------|---------------|------|
| `messages` | `messages` | JSON 序列化的消息列表 |
| `data_source` | `"vibe_rl_stock"` | 固定标识 |

---

### 6. RewardSignal（奖励信号）

对单条 AgentTrajectory 的综合打分，用于 GRPO/NGRPO 训练。

```python
@dataclass
class RewardSignal:
    trajectory_id: str     # 关联的轨迹 ID
    tool_correctness: float  # 工具调用正确性得分 [0, 0.4]
    output_quality: float    # 最终输出质量得分 [0, 0.4]（LLM-as-Judge 或规则）
    efficiency_penalty: float  # 效率惩罚 [-0.2, 0]（多余轮次扣分）
    final_reward: float    # 综合奖励 [0, 1]（clip 后）

    # 详细日志（用于调试）
    expected_tools: List[str]  # 预期调用的工具序列
    actual_tools: List[str]    # 实际调用的工具序列
    tool_match_detail: List[bool]  # 每次工具调用是否正确
```

**奖励计算公式**:
```
final_reward = clip(tool_correctness + output_quality + efficiency_penalty, 0, 1)
```

---

### 7. DebugRecord（调试记录）

HTTP 代理捕获的单次 LLM API 请求/响应记录，存储在 SQLite。

```python
@dataclass
class DebugRecord:
    record_id: str         # 唯一 ID（UUID）
    timestamp: str         # ISO 8601 时间戳
    request_id: Optional[str]  # LLM API 返回的请求 ID
    request_body: str      # JSON 格式的请求体
    response_body: str     # JSON 格式的响应体
    tool_calls: Optional[str]  # 提取的工具调用（JSON）
    duration_ms: int       # 请求耗时（毫秒）
    status_code: int       # HTTP 状态码
    error: Optional[str]   # 错误信息（如有）
```

**SQLite 表结构**:
```sql
CREATE TABLE debug_records (
    record_id    TEXT PRIMARY KEY,
    timestamp    TEXT NOT NULL,
    request_id   TEXT,
    request_body TEXT NOT NULL,
    response_body TEXT NOT NULL,
    tool_calls   TEXT,
    duration_ms  INTEGER NOT NULL,
    status_code  INTEGER NOT NULL,
    error        TEXT,
    -- 索引
    INDEX idx_timestamp (timestamp),
    INDEX idx_request_id (request_id)
);
```

---

### 8. TrainingConfig（训练配置）

所有训练超参数的配置对象，持久化为 YAML 文件，不允许硬编码。

```python
@dataclass
class TrainingConfig:
    # 模型配置
    model_path: str            # HuggingFace 模型路径或本地路径
    output_dir: str            # checkpoint 输出目录

    # FSDP 配置
    fsdp_param_offload: bool   # Actor 固定为 False，Ref 固定为 True
    fsdp_optimizer_offload: bool  # 默认 True（节省显存）

    # 训练超参
    num_train_epochs: int      # 训练 epoch 数（SFT: 3，GRPO: 5）
    batch_size: int            # 全局 batch size
    per_device_batch_size: int # 单卡 batch size
    gradient_accumulation_steps: int
    learning_rate: float       # SFT: 2e-5，GRPO: 5e-7
    gradient_checkpointing: bool  # MUST 为 True

    # GRPO/NGRPO 专用
    group_size: int            # 每个 prompt 采样数（默认 8）
    kl_coef: float             # KL 散度系数（默认 0.01）
    clip_ratio: float          # PPO clip 比率（默认 0.2）
    temperature: float         # 采样温度（默认 0.7）
    use_ngrpo: bool            # 是否启用 NGRPO（默认 False）
    ngrpo_threshold: float     # 切换 NGRPO 的方差阈值（默认 0.3）

    # 实验追踪
    wandb_project: str         # wandb 项目名
    wandb_run_name: str        # 实验运行名称
    seed: int                  # 随机种子（默认 42）
```

---

## 实体关系图

```
用户查询
   │
   ▼
AgentMessage (role=user)
   │
   ▼
[ReAct 循环]
   │
   ├─── AgentMessage (role=assistant, tool_calls=[ToolCall...])
   │         │
   │         ▼
   │    StockTool.handler() → 实际数据
   │         │
   │         ▼
   │    AgentMessage (role=tool, content=工具返回)
   │         │
   │    [下一轮...]
   │
   ▼
AgentTrajectory
   │
   ├─── RewardSignal (用于 GRPO 训练)
   └─── SFTSample (用于 SFT 数据生成)

DebugRecord ← HTTP 代理捕获每次 LLM API 调用
TrainingConfig → 持久化为 YAML，驱动所有训练脚本
```

---

## 状态转换（Agent 循环状态机）

```
INIT
  │ receive(user_query)
  ▼
REASONING  ─── LLM 生成 Thought + Action ───────────────────┐
  │                                                          │
  │ action.is_tool_call = True                               │
  ▼                                                          │
TOOL_CALLING ─── execute_tool() ─── WAITING_RESULT          │
  │                                      │                  │
  │ tool_result received                 │ timeout/error    │
  ▼                                      ▼                  │
PROCESSING ────────────── ERROR_HANDLING ──► REASONING      │
  │                                                         │
  │ action.is_final_answer = True / max_steps reached       │
  ▼                                                         │
COMPLETED ◄──────────────────────────────────────────────── ┘
```
