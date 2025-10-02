import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(base_model: str, lora_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()
    return tokenizer, model


def generate(prompt: str, tokenizer, model, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def main() -> None:
    base = os.getenv("BASE_MODEL", "bigcode/starcoder2-3b")
    lora_dir = os.getenv("LORA_DIR", "outputs/lora-codellama")
    prompt = os.getenv("PROMPT", "Write a Python function to add two numbers")
    tok, model = load_model(base, lora_dir)
    print(generate(prompt, tok, model))


if __name__ == "__main__":
    main()


