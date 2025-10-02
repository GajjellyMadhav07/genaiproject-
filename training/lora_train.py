import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class TrainConfig:
    base_model: str = os.getenv("BASE_MODEL", "bigcode/starcoder2-3b")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs/lora-codellama")
    dataset: str = os.getenv("DATASET", "code_search_net")  # or custom path
    text_field: str = os.getenv("TEXT_FIELD", "func_code_string")
    train_split: str = os.getenv("TRAIN_SPLIT", "train")
    eval_split: str = os.getenv("EVAL_SPLIT", "validation")
    seed: int = int(os.getenv("SEED", 42))
    lr: float = float(os.getenv("LR", 2e-4))
    epochs: int = int(os.getenv("EPOCHS", 1))
    per_device_train_batch_size: int = int(os.getenv("BATCH_SIZE", 1))
    gradient_accumulation_steps: int = int(os.getenv("GRAD_ACC", 8))
    lora_r: int = int(os.getenv("LORA_R", 8))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", 16))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", 0.05))
    max_length: int = int(os.getenv("MAX_LEN", 1024))


def tokenize_function(examples, tokenizer: AutoTokenizer, text_field: str, max_length: int):
    texts = examples[text_field]
    return tokenizer(texts, truncation=True, max_length=max_length)


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    dataset = load_dataset(cfg.dataset)
    train_ds = dataset[cfg.train_split]
    eval_ds = dataset[cfg.eval_split] if cfg.eval_split in dataset else None

    def _tok_fn(ex):
        return tokenize_function(ex, tokenizer, cfg.text_field, cfg.max_length)

    train_tok = train_ds.map(_tok_fn, batched=True, remove_columns=train_ds.column_names)
    eval_tok = (
        eval_ds.map(_tok_fn, batched=True, remove_columns=eval_ds.column_names)
        if eval_ds is not None
        else None
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="steps" if eval_tok is not None else "no",
        eval_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()


