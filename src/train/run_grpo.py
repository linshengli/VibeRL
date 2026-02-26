"""
GRPO / NGRPO 训练入口脚本

这是一个 standalone 的训练脚本，不依赖 verl 的内部 trainer。
适用于：
  1. 想从头理解 GRPO 训练流程的学习者
  2. verl API 变动时的 fallback
  3. 调试 reward 函数

如果 verl 的 main_grpo 可以直接用，推荐用：
  python -m verl.trainer.main_grpo --config configs/verl_grpo.yaml

本脚本用法：
  python src/train/run_grpo.py \
    --model-path checkpoints/sft/final \
    --prompts-file data/rl/prompts.parquet \
    --output-dir checkpoints/grpo \
    --group-size 8 \
    --use-ngrpo

注意：本脚本需要 GPU 环境和完整的训练依赖。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_prompts(prompts_file: str) -> List[str]:
    """从 parquet 或 jsonl 加载 prompt 列表"""
    path = Path(prompts_file)

    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df["prompt"].tolist()
    elif path.suffix in (".jsonl", ".json"):
        prompts = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict) and "prompt" in item:
                    prompts.append(item["prompt"])
                elif isinstance(item, str):
                    prompts.append(item)
        return prompts
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO / NGRPO standalone training script")
    parser.add_argument("--model-path", required=True, help="SFT checkpoint 路径")
    parser.add_argument("--prompts-file", required=True, help="RL prompt 数据文件")
    parser.add_argument("--output-dir", default="checkpoints/grpo", help="输出目录")
    parser.add_argument("--group-size", type=int, default=8, help="每个 prompt 采样数")
    parser.add_argument("--num-epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompt batch size")
    parser.add_argument("--lr", type=float, default=5e-7, help="学习率")
    parser.add_argument("--kl-coef", type=float, default=0.01, help="KL 系数")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--use-ngrpo", action="store_true", help="使用 NGRPO")
    parser.add_argument("--ngrpo-virtual-max", type=float, default=1.0, help="NGRPO 虚拟满分")
    parser.add_argument("--wandb-project", default="vibe-rl", help="Wandb project")
    parser.add_argument("--wandb-run", default="grpo-standalone", help="Wandb run name")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # ---- 检查依赖 ----
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("缺少训练依赖，请安装: pip install torch transformers")
        sys.exit(1)

    # ---- 加载 prompts ----
    prompts = load_prompts(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts")

    # ---- 加载模型 ----
    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ---- 导入 reward 和 NGRPO ----
    from src.train.rl_reward import AgentRewardComputer, RewardConfig
    from src.train.ngrpo import compute_batch_advantages

    reward_computer = AgentRewardComputer(RewardConfig())

    # ---- 训练循环 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 wandb（可选）
    try:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        use_wandb = True
    except Exception:
        use_wandb = False
        logger.info("Wandb not available, skipping logging")

    total_steps = 0
    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        # 按 batch 遍历 prompts
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_start:batch_start + args.batch_size]

            # ---- Step 1: Rollout（采样 group_size 个回答）----
            all_rewards = []
            all_log_probs = []

            for prompt in batch_prompts:
                group_rewards = []
                group_log_probs = []

                for _ in range(args.group_size):
                    # 生成回答
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            temperature=args.temperature,
                            top_p=0.95,
                            do_sample=True,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    # 计算 reward
                    reward = reward_computer.compute(prompt, response_text)
                    group_rewards.append(reward)

                    # 计算 log_prob（简化版，实际 verl 有更精确的实现）
                    with torch.no_grad():
                        full_ids = outputs.sequences[0].unsqueeze(0)
                        model_out = model(full_ids, labels=full_ids)
                        log_prob = -model_out.loss.item()
                    group_log_probs.append(log_prob)

                all_rewards.append(group_rewards)
                all_log_probs.append(group_log_probs)

            # ---- Step 2: 计算 Advantage ----
            rewards_tensor = torch.tensor(
                [r for group in all_rewards for r in group],
                dtype=torch.float32,
            )

            advantages, stds, ngrpo_count = compute_batch_advantages(
                rewards_tensor,
                group_size=args.group_size,
                use_ngrpo=args.use_ngrpo,
                virtual_max_reward=args.ngrpo_virtual_max,
            )

            # ---- Step 3: Policy Gradient 更新 ----
            # 简化版 —— 实际 verl 内部用的是完整的 PPO/GRPO 更新
            # 这里只做演示，展示 advantage-weighted policy gradient
            optimizer.zero_grad()

            # 对每个 prompt-response 对计算 loss
            loss_total = torch.tensor(0.0, requires_grad=True)
            if torch.cuda.is_available():
                loss_total = loss_total.cuda()

            idx = 0
            for prompt_idx, prompt in enumerate(batch_prompts):
                for resp_idx in range(args.group_size):
                    adv = advantages[idx].item()
                    # 简化: 用 advantage 作为 loss 的权重
                    # 实际应该用 ratio * advantage 并 clip
                    if abs(adv) > 1e-6:
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                        out = model(**inputs, labels=inputs["input_ids"])
                        loss_total = loss_total - adv * out.loss
                    idx += 1

            if loss_total.requires_grad:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_steps += 1

            # ---- Logging ----
            reward_mean = rewards_tensor.mean().item()
            reward_std = rewards_tensor.std().item()
            frac_zero_std = sum(1 for s in stds if s < 0.01) / len(stds) if len(stds) > 0 else 0

            log_dict = {
                "epoch": epoch + 1,
                "step": total_steps,
                "reward_mean": round(reward_mean, 4),
                "reward_std": round(reward_std, 4),
                "frac_reward_zero_std": round(frac_zero_std, 4),
                "ngrpo_count": ngrpo_count,
                "loss": round(loss_total.item(), 4) if hasattr(loss_total, 'item') else 0,
            }

            if total_steps % 10 == 0:
                logger.info(json.dumps(log_dict, ensure_ascii=False))

            if use_wandb:
                wandb.log(log_dict, step=total_steps)

        # ---- Epoch 结束，保存 checkpoint ----
        ckpt_dir = output_dir / f"epoch_{epoch + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # 保存最终模型
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Training complete. Final model: {final_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
