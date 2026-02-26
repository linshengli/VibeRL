from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.train.common import extract_last_assistant_answer, parse_messages_payload


@dataclass
class SFTScriptArgs:
    model_path: str
    train_parquet: str
    val_parquet: str
    output_dir: str
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200


def _require_ml_stack():
    try:
        import pandas as pd  # noqa: F401
        import torch  # noqa: F401
        from datasets import Dataset  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SFT 训练依赖缺失，请安装: pip install torch transformers datasets accelerate pandas pyarrow"
        ) from exc


def _render_prompt_from_messages(tokenizer: Any, messages: List[Dict[str, Any]], target: str) -> str:
    context = []
    seen_target = False
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role == "assistant" and content == target and not seen_target:
            seen_target = True
            break
        if role in {"system", "user", "assistant"} and content:
            context.append({"role": role, "content": content})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    lines: List[str] = []
    for item in context:
        lines.append(f"{item['role']}: {item['content']}")
    lines.append("assistant:")
    return "\n".join(lines)


def _load_parquet_records(path: str) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    records = df.to_dict(orient="records")
    return [x for x in records if isinstance(x, dict)]


def _build_supervised_examples(tokenizer: Any, rows: List[Dict[str, Any]], max_length: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    eos = tokenizer.eos_token or ""

    for row in rows:
        messages = parse_messages_payload(row.get("messages"))
        if not messages:
            continue
        target = extract_last_assistant_answer(messages)
        if not target:
            continue
        prompt = _render_prompt_from_messages(tokenizer, messages, target)

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_text = f"{prompt}{target}{eos}"
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
            if len(prompt_ids) >= len(full_ids):
                continue
            # Align prompt length after left truncation.
            prompt_ids = prompt_ids[-(len(prompt_ids) - (len(tokenizer(prompt, add_special_tokens=False).input_ids) - len(full_ids))):]

        labels = [-100] * min(len(prompt_ids), len(full_ids))
        labels += full_ids[len(labels) :]
        labels = labels[: len(full_ids)]
        attention_mask = [1] * len(full_ids)

        examples.append(
            {
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )
    return examples


def run_sft(args: SFTScriptArgs) -> None:
    _require_ml_stack()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_rows = _load_parquet_records(args.train_parquet)
    val_rows = _load_parquet_records(args.val_parquet)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = _build_supervised_examples(tokenizer, train_rows, max_length=args.max_length)
    val_examples = _build_supervised_examples(tokenizer, val_rows, max_length=args.max_length)
    if not train_examples:
        raise RuntimeError("train set 没有可用样本，请检查 parquet 中的 messages 字段")

    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples or train_examples[:16])

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        pad_id = int(tokenizer.pad_token_id)
        for item in batch:
            pad_n = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_n)
            attention_mask.append(item["attention_mask"] + [0] * pad_n)
            labels.append(item["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )
    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    print(f"SFT 完成，模型已保存到: {out / 'final'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local SFT training with HF Trainer")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=200)
    ns = parser.parse_args()

    args = SFTScriptArgs(
        model_path=ns.model_path,
        train_parquet=ns.train_parquet,
        val_parquet=ns.val_parquet,
        output_dir=ns.output_dir,
        learning_rate=ns.learning_rate,
        num_train_epochs=ns.num_train_epochs,
        per_device_train_batch_size=ns.per_device_train_batch_size,
        per_device_eval_batch_size=ns.per_device_eval_batch_size,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        max_length=ns.max_length,
        logging_steps=ns.logging_steps,
        eval_steps=ns.eval_steps,
        save_steps=ns.save_steps,
    )
    run_sft(args)


if __name__ == "__main__":
    main()

