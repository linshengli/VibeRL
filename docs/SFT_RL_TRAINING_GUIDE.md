# SFT / RL 训练落地指南（当前仓库）

本文给你一个务实路线：先跑通当前仓库已实现能力，再进入 verl 的 SFT/RL 训练。

## 1. 先确认现状（很关键）

当前仓库**已实现**：
- SFT 数据生成：`src/data/generate_sft.py`
- 数据校验：`src/data/validate.py`
- 转 parquet：`src/data/to_parquet.py`
- 奖励函数：`src/reward/reward_computer.py`
- 评估脚本：`src/eval/evaluate.py` / `src/eval/compare.py`

当前仓库**未实现**（需要你补齐）：
- verl 的训练配置文件（`configs/sft_config.yaml`、`configs/grpo_config.yaml` 不在仓库）
- RL rollout/reward 对接脚本（给 verl 用的在线 reward 管线）
- 本地 checkpoint 推理接入当前 `AgentEvaluator`（`evaluate.py` 没有 `base_url` 参数）

结论：你现在可以完整跑“数据准备 + 基线评估”，但要进入 SFT/RL 训练需先补训练配置与推理对接。

## 2. 当前仓库可直接执行的流程

### 2.1 环境与测试

```bash
cd /Users/tbxsx/Code/VibeRL
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pytest -q
```

### 2.2 生成 SFT 样本

```bash
python src/data/generate_sft.py \
  --output-dir data/sft \
  --num-samples 1000 \
  --model deepseek-chat \
  --market-dist '{"a_share": 0.4, "hk_share": 0.3, "mixed": 0.3}' \
  --seed 42
```

### 2.3 校验 + 转 parquet

```bash
python src/data/validate.py --data-dir data/sft

python src/data/to_parquet.py \
  --input data/sft/samples.jsonl \
  --output-dir data/sft/parquet \
  --split 0.9
```

目标产物：
- `data/sft/samples.jsonl`
- `data/sft/parquet/train.parquet`
- `data/sft/parquet/val.parquet`

### 2.4 基线评估（训练前）

```bash
python src/eval/evaluate.py \
  --model deepseek-chat \
  --test-cases data/test_cases.json \
  --output results/baseline_eval.json
```

## 3. 继续做 SFT / RL 的最短路径

### 步骤 A：安装训练栈（verl）

参考：`specs/001-vibe-coding-agent-rl/quickstart.md`

核心依赖：
- `verl`
- `vllm`
- `wandb`（可选）
- `torch`（CUDA 对应版本）

### 步骤 B：补齐训练配置文件

你需要新增：
- `configs/sft_config.yaml`
- `configs/grpo_config.yaml`

数据输入使用：
- `data/sft/parquet/train.parquet`
- `data/sft/parquet/val.parquet`

### 步骤 C：执行训练

SFT（示意命令）：
```bash
torchrun --nproc_per_node=4 \
  -m verl.trainer.fsdp_sft_trainer \
  --config configs/sft_config.yaml
```

GRPO（示意命令）：
```bash
python -m verl.trainer.main_grpo \
  --config configs/grpo_config.yaml
```

NGRPO（低方差时）：
```bash
python -m verl.trainer.main_grpo \
  --config configs/grpo_config.yaml \
  --use_ngrpo true
```

## 4. 训练后评估怎么接

当前 `src/eval/evaluate.py` 只接收 `--model`，没有 `--base-url`。

如果你把 checkpoint 部署在 vLLM/OpenAI 兼容服务上，建议先补这两处：
- `src/eval/evaluate.py` 增加 `--base-url`
- `src/eval/evaluator.py` 透传给 `StockAnalysisAgent(model=..., base_url=...)`

否则直接用本地 checkpoint 路径，当前评估脚本不会真正加载该权重。

## 5. 推荐里程碑（按周）

1. 周 1：扩展数据集（从 1k 到 10k），加强工具调用链多样性
2. 周 2：补 `configs/*` + 跑通 1 epoch SFT（先看 loss 与工具准确率）
3. 周 3：接入 GRPO，重点监控 `reward_std` 与 `frac_reward_zero_std`
4. 周 4：A/B 报告（baseline vs SFT vs RL）

## 6. 你现在可以直接参考的文件

- 数据生成逻辑：[generator.py](/Users/tbxsx/Code/VibeRL/src/data/generator.py)
- 奖励函数：[reward_computer.py](/Users/tbxsx/Code/VibeRL/src/reward/reward_computer.py)
- 评估流程：[evaluator.py](/Users/tbxsx/Code/VibeRL/src/eval/evaluator.py)
- 详细 quickstart 草案：[quickstart.md](/Users/tbxsx/Code/VibeRL/specs/001-vibe-coding-agent-rl/quickstart.md)

