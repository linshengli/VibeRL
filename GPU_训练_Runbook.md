# GPU 训练 Runbook：从零到跑通 SFT + RL 全流程

> 本文档基于对当前仓库代码的逐文件审计，列出了「已完成」与「待补齐」的完整清单。
> 你只需按这个文档，从上往下执行即可。

---

## 0. 当前仓库代码审计结论

### ✅ 已完成（可直接用）

| 模块 | 文件 | 说明 |
|------|------|------|
| 单 Agent（ReAct） | `src/agent/core.py` | 支持 rule-based 回退 + OpenAI API 调用，多轮工具调用完整 |
| 4 个股票工具 | `src/tools/stock_tools.py` | search/info/quote/indicators，真实接 akshare API |
| Multi-Agent 编排器 | `src/agentic/orchestrator.py` | 状态驱动，支持 Planner/Search/Quote/Reasoning/Risk/Reflect 多 Agent |
| 数据校验 | `src/data/validate.py` | 检查 JSONL 字段完整性 |
| Parquet 转换 | `src/data/to_parquet.py` | JSONL → Parquet（verl 格式） |
| 奖励函数 | `src/reward/reward_computer.py` | 工具准确性 + 输出质量 + 效率惩罚 |
| 评估框架 | `src/eval/evaluate.py` + `evaluator.py` | 跑 Agent 并算 tool_accuracy / output_quality / amb_accuracy |
| 调试代理 | `src/debugger/proxy.py` | HTTP 代理，记录 API 请求到 SQLite |
| 实体定义 | `src/models/entities.py` | 所有 dataclass 齐全 |
| 配置文件 | `config/config.yaml` | 基础参数（有重复 key 需修） |

### ❌ 待补齐（你在 GPU 机器上需要写的）

| 编号 | 缺失项 | 严重程度 | 说明 |
|------|--------|----------|------|
| **D1** | SFT 数据生成器是假的 | 🔴 致命 | `generator.py` 没有调 DeepSeek API，只是拼接硬编码模板，生成的 messages 里没有真正的 tool_calls 字段 |
| **D2** | 只有 7 个 prompt 模板 | 🔴 致命 | `generate_sft.py` 中 PROMPT_POOL 只有 7 条，循环复用生成 1000 条完全重复 |
| **D3** | 测试集太小 | 🟡 重要 | `data/test_cases.json` 仅 5 条用例 |
| **D4** | SFT 脚本不是 verl 的 | 🟡 重要 | `src/train/sft.py` 用的是 HF Trainer，不是 verl 的 FSDP SFT Trainer |
| **D5** | 没有 verl SFT 配置 | 🔴 致命 | 不存在 `configs/sft_config.yaml` |
| **D6** | 没有 RL 训练脚本 | 🔴 致命 | 整个仓库没有任何 GRPO/NGRPO 训练代码 |
| **D7** | 没有 verl GRPO 配置 | 🔴 致命 | 不存在 `configs/grpo_config.yaml` |
| **D8** | 没有 RL Rollout 环境 | 🔴 致命 | 没有给 verl 写 Agent 环境（让模型在线调工具、拿 reward） |
| **D9** | 评估脚本不支持本地模型 | 🟡 重要 | `evaluate.py` 没有 `--base-url` 参数，无法评估 vLLM 部署的 checkpoint |
| **D10** | config.yaml 有重复 key | 🟢 小问题 | `num_train_epochs` 和 `gradient_checkpointing` 各出现两次 |
| **D11** | reward 不支持 LLM-as-Judge | 🟡 重要 | `use_llm_judge` 参数存在但未实现（直接被忽略） |

---

## 1. 租 GPU 与环境搭建

### 1.1 硬件选择

| 方案 | 配置 | 适用阶段 | 估算日租金 |
|------|------|----------|------------|
| 经济型 | 2× L4 (24GB) | SFT only | ¥50-80 |
| **推荐** | **4× L4 (24GB)** | **SFT + GRPO** | **¥120-200** |
| 充裕型 | 2× A100 (80GB) | 全流程无忧 | ¥300-500 |

> ⚠️ 博文作者双 L4 跑不起来（没开 gradient checkpointing），4 卡 L4 可以运行。
> 建议直接上 4× L4，省得反复迁移。

平台推荐：AutoDL、硅基流动、Vast.ai、Lambda Cloud

### 1.2 环境安装（按顺序执行）

```bash
# ---- 1. 基础环境 ----
conda create -n viberl python=3.10 -y
conda activate viberl

# ---- 2. PyTorch (CUDA 12.1) ----
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ---- 3. verl 框架 ----
pip install verl

# ---- 4. vLLM（推理引擎，verl rollout 依赖）----
pip install vllm

# ---- 5. 项目依赖 ----
cd ~/VibeRL
pip install -r requirements.txt
pip install wandb transformers datasets accelerate pyarrow

# ---- 6. 下载模型 ----
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

# ---- 7. 验证 ----
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
python -c "import verl; print('verl OK')"
python -c "import vllm; print('vllm OK')"
pytest -q  # 跑项目单元测试
```

---

## 2. 修复 D10：config.yaml 重复 key

**文件：** `config/config.yaml`

去掉重复的 `num_train_epochs` 和 `gradient_checkpointing`，保留一份即可。建议拆分为两个配置：

```yaml
# config/sft_config.yaml
model_path: models/Qwen2.5-7B-Instruct
train_files: data/sft/parquet/train.parquet
val_files: data/sft/parquet/val.parquet
output_dir: checkpoints/sft

num_train_epochs: 3
per_device_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-5
gradient_checkpointing: true
fsdp_param_offload: false
fsdp_optimizer_offload: true

wandb_project: vibe-rl
wandb_run_name: sft-v1
seed: 42
```

```yaml
# config/grpo_config.yaml
model_path: checkpoints/sft/final
ref_model_path: checkpoints/sft/final
output_dir: checkpoints/grpo

# GRPO 超参
group_size: 8
kl_coef: 0.01
clip_ratio: 0.2
temperature: 0.7

# NGRPO（reward 方差低时开启）
use_ngrpo: false
ngrpo_virtual_max_reward: 1.0

# FSDP - Actor（不 offload，频繁更新）
actor_param_offload: false
actor_optimizer_offload: false

# FSDP - Ref Model（offload，不更新）
ref_param_offload: true
ref_optimizer_offload: true

total_epochs: 5
batch_size: 64
learning_rate: 5e-7
gradient_checkpointing: true

wandb_project: vibe-rl
wandb_run_name: grpo-v1
seed: 42
```

---

## 3. 修复 D1 + D2：重写 SFT 数据生成器

**问题：** 当前 `generator.py` 根本没调 DeepSeek API，只是拼接了三行固定文本，messages 里没有 tool_calls 字段。

**你需要做的：**

### 3.1 扩充 prompt 池

在 `src/data/generate_sft.py` 中，把 PROMPT_POOL 从 7 条扩到至少 **100 条**，覆盖：

```python
PROMPT_POOL = [
    # ---- A 股基础 ----
    "帮我查一下茅台今天的行情",
    "看看中国平安最新的价格",
    "查一下比亚迪的技术指标",
    "分析一下宁德时代",
    # ...

    # ---- 港股基础 ----
    "帮我查一下腾讯的股价",
    "看看美团的行情",
    "小米集团最近表现怎么样",
    # ...

    # ---- 混淆场景（核心！）----
    "查一下平安的股票",           # 中国平安 vs 平安银行 vs 平安好医生(港股)
    "帮我看看阿里巴巴",           # A 股 vs 港股
    "比较中国平安和平安银行",      # 同名歧义
    "对比腾讯和茅台",             # 跨市场
    # ...

    # ---- 多只股票 ----
    "比较茅台、五粮液和泸州老窖",
    "看看 BAT 三家的行情",
    # ...

    # ---- 指定代码 ----
    "查一下 600519 的行情",
    "00700 最近怎么样",
    "帮我看 300750 的 MACD",
    # ...
]
```

### 3.2 改写 generator.py —— 真正调 DeepSeek 生成多轮对话

核心改造：让 DeepSeek API 扮演 Agent，执行真实的多轮 tool calling，记录完整轨迹。

```python
# src/data/generator_real.py（新文件）

import json
import uuid
from openai import OpenAI
from src.tools.stock_tools import get_tool_definitions, get_stock_tool_registry

class RealSFTDataGenerator:
    """用 DeepSeek API 真正执行 Agent 循环，生成高质量 SFT 数据"""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.tools = get_stock_tool_registry()
        self.tool_defs = get_tool_definitions()

    def generate_one(self, prompt: str, max_turns: int = 10) -> dict:
        """执行一次完整的 Agent 循环，返回 SFT 训练样本"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_turns):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_defs,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                # 记录 assistant 的 tool_calls
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                }
                messages.append(assistant_msg)

                # 执行工具，把结果加回去
                for tc in msg.tool_calls:
                    tool_fn = self.tools.get(tc.function.name)
                    if tool_fn:
                        args = json.loads(tc.function.arguments)
                        result = tool_fn.handler(**args)
                    else:
                        result = {"error": f"Unknown tool: {tc.function.name}"}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
            else:
                # 最终回答
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                })
                break

        return {
            "sample_id": str(uuid.uuid4()),
            "messages": messages,
            "prompt": prompt,
        }
```

### 3.3 生成 1000 条真实数据

```bash
export DEEPSEEK_API_KEY="sk-your-key"

python src/data/generate_sft.py \
  --output-dir data/sft \
  --num-samples 1000 \
  --model deepseek-chat \
  --market-dist '{"a_share": 0.4, "hk_share": 0.3, "mixed": 0.3}' \
  --seed 42

# 校验
python src/data/validate.py --data-dir data/sft

# 转 parquet
python src/data/to_parquet.py \
  --input data/sft/samples.jsonl \
  --output-dir data/sft/parquet \
  --split 0.9
```

**预期产出：**
- `data/sft/samples.jsonl` —— 1000 条含真实 tool_calls 的多轮对话
- `data/sft/parquet/train.parquet` —— 900 条训练集
- `data/sft/parquet/val.parquet` —— 100 条验证集

**DeepSeek API 费用估算：** 1000 条 × ~5 轮 × ~2000 tokens ≈ 10M tokens ≈ ¥10-20

---

## 4. 修复 D3：扩充测试集

当前只有 5 条测试用例，至少需要 **50-100 条**。

```bash
# 建议结构
data/test_cases.json  # 扩到 50-100 条，覆盖：
  # - 15 条纯 A 股
  # - 15 条纯港股
  # - 10 条 A/港混淆（is_ambiguous: true）
  # - 5 条多只股票对比
  # - 5 条指定代码查询
  # - 5 条技术指标查询
  # - 5 条边界 case（不存在的股票、无效代码等）
```

每条测试用例格式：
```json
{
  "query": "查一下中芯国际的行情",
  "expected_tools": ["search_stock_by_name", "get_realtime_quote"],
  "ground_truth": {"market": "a_share", "code": "688981"},
  "is_ambiguous": false,
  "category": "a_share_basic"
}
```

---

## 5. 修复 D4 + D5：接入 verl SFT 训练

### 方案选择

当前 `src/train/sft.py` 用的是 HF Trainer，**可以用但效率低**。两个选择：

| 方案 | 优点 | 缺点 |
|------|------|------|
| A. 继续用 HF Trainer + FSDP | 已有代码，改动小 | 需自己加 FSDP 配置 |
| B. 切换 verl FSDP SFT Trainer | 原生支持 FSDP/offload，跟 RL 阶段无缝衔接 | 需新写配置 |

**推荐方案 B**，这样 SFT 和 RL 在同一框架里，checkpoint 格式兼容。

### 5.1 创建 verl SFT 配置

参考 verl 的 examples/sft 目录，创建 `configs/verl_sft.yaml`：

```yaml
# configs/verl_sft.yaml
# 参考: https://github.com/volcengine/verl/tree/main/examples/sft

model:
  path: models/Qwen2.5-7B-Instruct

data:
  train_files: data/sft/parquet/train.parquet
  val_files: data/sft/parquet/val.parquet
  max_length: 4096  # 多轮对话需要更长

training:
  output_dir: checkpoints/sft
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  warmup_ratio: 0.03
  weight_decay: 0.01
  bf16: true
  gradient_checkpointing: true  # ⚠️ 必须开启！

fsdp:
  enabled: true
  param_offload: true    # 4×L4 建议开启
  optimizer_offload: true

logging:
  wandb_project: vibe-rl
  wandb_run_name: sft-v1
  logging_steps: 10

save:
  save_steps: 200
  save_total_limit: 3
```

### 5.2 执行 SFT 训练

```bash
# 使用 verl
torchrun --nproc_per_node=4 \
  -m verl.trainer.fsdp_sft_trainer \
  --config configs/verl_sft.yaml

# 或者继续用 HF Trainer（已有代码）
python -m src.train.sft \
  --model-path models/Qwen2.5-7B-Instruct \
  --train-parquet data/sft/parquet/train.parquet \
  --val-parquet data/sft/parquet/val.parquet \
  --output-dir checkpoints/sft \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 3 \
  --max-length 4096
```

**预期耗时：** 4×L4，1000 条数据，3 epochs ≈ 1-2 小时

### 5.3 SFT 后评估

```bash
# 先用 vLLM 部署 SFT checkpoint
python -m vllm.entrypoints.openai.api_server \
  --model checkpoints/sft/final \
  --port 8000

# 跑评估（需要先修复 D9，见下文）
python src/eval/evaluate.py \
  --model checkpoints/sft/final \
  --base-url http://localhost:8000/v1 \
  --test-cases data/test_cases.json \
  --output results/sft_eval.json
```

---

## 6. 修复 D9：评估脚本支持本地模型

**改动点：**

`src/eval/evaluate.py` 增加 `--base-url` 参数：

```python
parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
```

`src/eval/evaluator.py` 的 `AgentEvaluator.evaluate()` 中，构造 Agent 时传入 base_url：

```python
agent = StockAnalysisAgent(model=model_name, base_url=base_url)
```

`src/agent/core.py` 已经支持 `base_url` 参数，不需要改。

---

## 7. 修复 D6 + D7 + D8：RL 训练全套

这是最大的工作量。需要写三样东西：

### 7.1 RL Rollout 环境

verl 需要一个环境，让模型在线 rollout（生成回答），然后计算 reward。

```python
# src/train/rl_environment.py（新文件）

"""
verl 的 GRPO 训练需要：
1. 给定 prompt，让 actor 模型生成 group_size 个回答
2. 对每个回答计算 reward
3. 把 (prompt, response, reward) 喂给训练器

对于 Agent 场景，"生成回答"不只是一次 generate：
- 模型可能会输出 tool_calls
- 需要执行工具拿结果
- 再让模型继续生成
- 如此循环直到输出最终回答
"""

import json
from typing import List, Dict, Any
from src.tools.stock_tools import get_stock_tool_registry
from src.reward.reward_computer import RewardComputer


class AgentRolloutEnvironment:
    """
    Agent RL 环境：执行多轮 tool calling，返回完整轨迹 + reward
    """

    def __init__(self, test_cases: List[Dict], max_turns: int = 10):
        self.tools = get_stock_tool_registry()
        self.reward_computer = RewardComputer()
        self.test_cases = test_cases
        self.max_turns = max_turns

    def get_prompts(self) -> List[str]:
        """返回所有训练 prompt"""
        return [case["query"] for case in self.test_cases]

    def compute_reward(self, prompt: str, response_messages: List[Dict]) -> float:
        """
        给定 prompt 和模型生成的完整对话，计算 reward

        这里需要：
        1. 解析 response_messages 中的 tool_calls
        2. 执行工具拿到结果
        3. 跟 expected_tools 对比
        4. 评估最终输出质量
        """
        # 找到对应的 test case
        case = None
        for c in self.test_cases:
            if c["query"] == prompt:
                case = c
                break

        expected_tools = case.get("expected_tools", []) if case else []

        # 从 response 中提取实际的 tool calls
        actual_tools = []
        final_output = ""
        for msg in response_messages:
            if msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        actual_tools.append(tc["function"]["name"])
                elif msg.get("content"):
                    final_output = msg["content"]

        # 计算 reward（复用现有的 RewardComputer 逻辑）
        # tool_correctness: 0-0.4
        tool_score = 0.0
        if expected_tools:
            matched = sum(1 for a, e in zip(actual_tools, expected_tools) if a == e)
            tool_score = min(0.4, (matched / len(expected_tools)) * 0.4)
        elif actual_tools:
            tool_score = 0.4

        # output_quality: 0-0.4
        output_score = 0.0
        if final_output and len(final_output) >= 20:
            output_score += 0.1
        if any(ch.isdigit() for ch in final_output):
            output_score += 0.1
        if any(kw in final_output for kw in ["价格", "涨跌", "行情"]):
            output_score += 0.1
        if "(" in final_output and ")" in final_output:
            output_score += 0.1

        # efficiency_penalty: -0.2 ~ 0
        num_turns = len([m for m in response_messages if m.get("role") == "assistant"])
        efficiency = 0.0
        if num_turns > 3:
            efficiency = -min(0.2, (num_turns - 3) * 0.02)

        return max(0.0, min(1.0, tool_score + output_score + efficiency))
```

### 7.2 verl GRPO 训练配置

创建 `configs/verl_grpo.yaml`：

```yaml
# configs/verl_grpo.yaml
# 参考: https://github.com/volcengine/verl/tree/main/examples/grpo
# 参考: https://github.com/volcengine/verl/tree/main/verl-recipe

actor_rollout_ref:
  model:
    path: checkpoints/sft/final   # SFT 后的 checkpoint

  actor:
    fsdp_config:
      param_offload: false         # ⚠️ Actor 不 offload！频繁更新
      optimizer_offload: false
    lr: 5e-7                       # RL 阶段学习率低
    gradient_checkpointing: true

  ref:
    fsdp_config:
      param_offload: true          # Ref model 可以 offload
      optimizer_offload: true

  rollout:
    # vLLM 配置（用于在线生成）
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.4
    temperature: 0.7
    top_p: 0.95
    max_new_tokens: 2048

algorithm:
  grpo:
    group_size: 8                  # 每个 prompt 采样 8 个回答
    kl_coef: 0.01
    clip_ratio: 0.2

    # NGRPO（如遇 reward_std 很低，改为 true）
    use_ngrpo: false
    ngrpo_virtual_max_reward: 1.0

trainer:
  total_epochs: 5
  micro_batch_size: 4
  gradient_accumulation_steps: 2
  save_steps: 100
  logging_steps: 10
  gradient_checkpointing: true

data:
  # RL prompt 数据集（只需 prompt，不需要 response）
  train_files: data/rl/prompts.parquet
  max_prompt_length: 512
  max_response_length: 2048

reward:
  # 自定义 reward 函数
  reward_fn: src.train.rl_environment:compute_reward

logging:
  wandb_project: vibe-rl
  wandb_run_name: grpo-v1
```

### 7.3 准备 RL prompt 数据

```python
# src/data/generate_rl_prompts.py（新文件）

"""
RL 训练不需要 response，只需要 prompt。
从扩充的 prompt 池 + 测试集生成 RL prompt 数据集。
"""

import pandas as pd

prompts = [
    "帮我查一下茅台今天的行情",
    "查一下腾讯的股价",
    "平安最近怎么样",
    # ... 扩到 200-500 条
]

df = pd.DataFrame({"prompt": prompts, "data_source": "vibe_rl_stock"})
df.to_parquet("data/rl/prompts.parquet", index=False)
```

### 7.4 NGRPO 实现

**这是关键**：当你发现训练中 `reward_std` 很低、`frac_reward_zero_std` 很高时需要切换。

需要在 verl 的 core_algos 中注入 NGRPO 逻辑。参考实现：
https://github.com/nangongrui-ngr/NGRPO/blob/ngr/verl/trainer/ppo/core_algos.py#L156

```python
# src/train/ngrpo.py（新文件）

import torch


def compute_ngrpo_advantages(
    rewards: torch.Tensor,          # shape: (group_size,)
    virtual_max_reward: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    NGRPO: 在每组 rewards 中添加虚拟满分样本，拉大组内方差。

    原理：
    - GRPO 里 advantage = (r - mean) / std
    - 如果组内所有回答都拿了 0.3 分，std ≈ 0，advantage ≈ 0，模型不学习
    - NGRPO 加一个虚拟的满分样本 (1.0)，强行拉大 std
    - 只对真实样本计算 advantage（虚拟样本不参与梯度更新）
    """
    augmented = torch.cat([rewards, torch.tensor([virtual_max_reward], device=rewards.device)])
    mean = augmented.mean()
    std = augmented.std()

    # 只对真实样本计算 advantage
    advantages = (rewards - mean) / (std + eps)
    return advantages
```

### 7.5 执行 RL 训练

```bash
# 准备 RL prompt 数据
python src/data/generate_rl_prompts.py

# 运行 GRPO
python -m verl.trainer.main_grpo \
  --config configs/verl_grpo.yaml

# 如果 reward_std 低，切 NGRPO
python -m verl.trainer.main_grpo \
  --config configs/verl_grpo.yaml \
  --algorithm.grpo.use_ngrpo true
```

**监控指标（wandb）：**
- `reward_mean` —— 应该逐步上升
- `reward_std` —— 不能接近 0
- `frac_reward_zero_std` —— 应该低于 0.3
- `kl_divergence` —— 不应该无限增长
- `policy_loss` —— 应该逐步下降

---

## 8. 修复 D11：增强 Reward 函数

当前 `reward_computer.py` 的 `_output_score` 用的是简单规则匹配。建议增加 LLM-as-Judge：

```python
# src/reward/llm_judge.py（新文件）

from openai import OpenAI

JUDGE_PROMPT = """
请评估以下股票分析 Agent 的回答质量，从 0 到 1 打分。

评分标准：
- 0.0-0.2: 回答为空或完全无关
- 0.2-0.4: 提到了股票但信息不完整
- 0.4-0.6: 有基本信息但分析粗糙
- 0.6-0.8: 信息完整，分析有理据
- 0.8-1.0: 信息全面，推理清晰，有实用价值

用户问题：{query}
Agent 回答：{answer}

请只返回一个数字（0到1之间的小数）。
"""

class LLMJudge:
    def __init__(self, model="deepseek-chat", api_key=None):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

    def score(self, query: str, answer: str) -> float:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(query=query, answer=answer)}],
            temperature=0.0,
        )
        try:
            return float(resp.choices[0].message.content.strip())
        except:
            return 0.0
```

---

## 9. 最终评估与对比

```bash
# 1. 部署三个版本
# 基线（直接用 Qwen2.5-7B-Instruct）
vllm serve models/Qwen2.5-7B-Instruct --port 8001

# SFT 版
vllm serve checkpoints/sft/final --port 8002

# RL 版
vllm serve checkpoints/grpo/final --port 8003

# 2. 跑评估
for port in 8001 8002 8003; do
  python src/eval/evaluate.py \
    --model "Qwen2.5-7B-Instruct" \
    --base-url "http://localhost:${port}/v1" \
    --test-cases data/test_cases.json \
    --output "results/eval_port${port}.json"
done

# 3. 对比
python src/eval/compare.py \
  --baseline results/eval_port8001.json \
  --sft results/eval_port8002.json \
  --rl results/eval_port8003.json \
  --output results/final_comparison.json
```

**预期结果：**

| 模型 | 工具准确率 | 输出质量 | A/港混淆 |
|------|-----------|---------|---------|
| Qwen2.5-7B (基线) | ~60% | ~50% | ~40% |
| + SFT | ~75% | ~65% | ~60% |
| + GRPO/NGRPO | ~85% | ~80% | ~80% |
| deepseek-chat (参考) | ~85% | ~85% | ~85% |

---

## 10. 完整执行顺序 Checklist

```
阶段一：环境（1 天）
  □ 1.  租 GPU 机器（推荐 4×L4）
  □ 2.  安装 conda + PyTorch + verl + vLLM
  □ 3.  clone 仓库，跑 pytest 确认通过
  □ 4.  下载 Qwen2.5-7B-Instruct
  □ 5.  获取 DeepSeek API key

阶段二：数据（2-3 天）
  □ 6.  修复 config.yaml 重复 key
  □ 7.  扩充 PROMPT_POOL 到 100+ 条
  □ 8.  重写 generator.py，真正调 DeepSeek API 生成 tool_calls 对话
  □ 9.  生成 1000 条 SFT 数据，校验，转 parquet
  □ 10. 扩充 test_cases.json 到 50-100 条
  □ 11. 跑基线评估（原始 Qwen2.5-7B + deepseek-chat）

阶段三：SFT（2-3 天）
  □ 12. 创建 verl SFT 配置文件
  □ 13. 执行 SFT 训练（3 epochs）
  □ 14. 修复 evaluate.py 增加 --base-url
  □ 15. vLLM 部署 SFT checkpoint，跑评估
  □ 16. 对比 SFT vs 基线，确认工具调用准确率提升

阶段四：RL（3-5 天）
  □ 17. 准备 RL prompt 数据集
  □ 18. 编写 RL rollout 环境（Agent 环境 + reward 计算）
  □ 19. 创建 verl GRPO 配置文件
  □ 20. 执行 GRPO 训练，监控 wandb
  □ 21. 如遇 reward_std 低，实现并切换 NGRPO
  □ 22. vLLM 部署 RL checkpoint，跑评估

阶段五：收尾（1-2 天）
  □ 23. 三版本对比（基线 vs SFT vs RL）
  □ 24. （可选）实现 LLM-as-Judge reward
  □ 25. （可选）尝试 DART 解耦训练解决推理/工具梯度冲突
  □ 26. 写总结报告

预计总计：9-14 天
```

---

## 附录 A：常见坑与排错

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| OOM | CUDA out of memory | 开 gradient_checkpointing + FSDP offload |
| SFT 后推理变弱 | 工具调用准，但反复确认 | 可能是推理/工具梯度冲突，参考 DART 论文 |
| reward_std ≈ 0 | frac_reward_zero_std 很高 | 切 NGRPO，或调整奖励函数拉大区分度 |
| Actor offload 误开 | 训练极慢 | 检查 actor.fsdp_config.param_offload 必须 false |
| vLLM 加载 checkpoint 失败 | 模型格式不兼容 | 确保 SFT 保存了完整模型（不是只保存 adapter） |
| akshare API 限流 | 数据生成报 429 | 加 retry + sleep，或分批次生成 |
| verl 版本不兼容 | import error | 锁定 verl 版本，pip install verl==0.6.x |

## 附录 B：关键参考资料

- verl GRPO 文档：https://verl.readthedocs.io/en/latest/algo/grpo.html
- verl-recipe 仓库：https://github.com/volcengine/verl-recipe
- NGRPO 参考实现：https://github.com/nangongrui-ngr/NGRPO
- 推理与工具梯度冲突论文：https://arxiv.org/abs/2602.00994
- GRPO 原论文：https://arxiv.org/abs/2402.03300

---

*文档生成日期：2026-02-26*
*基于仓库代码逐文件审计后生成*
