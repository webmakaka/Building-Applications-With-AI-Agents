import os
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.environ['HF_TOKEN']="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# fine_tune_function_calling.py
"""
针对 P40 (24GB) 与 RTX 2060 (6GB) 优化的微调脚本。
主要修改：适配 fp16 计算，修正量化参数，增强硬件兼容性。
"""
from __future__ import annotations

import argparse
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

###############################################################################
# Special tokens & chat template helpers
###############################################################################

class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_response>"
    eotool_response = "</tool_response>"
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def list(cls) -> List[str]:
        return [c.value for c in cls]

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content']|trim + '<end_of_turn><eos>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)

###############################################################################
# Dataset preprocessing
###############################################################################

def _merge_system_into_first_user(messages: List[Dict[str, str]]) -> None:
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        messages.pop(0)
        if not messages or (messages[0]["role"] not in ["human", "user"]):
            raise ValueError("Expected a user message after system message.")
        messages[0]["content"] = (
            f"{system_content}Also, before making a call to a function take the time "
            "to plan the function to take. Make that thinking process between "
            "<think>{your thoughts}</think>\n\n" + messages[0]["content"]
        )

def build_preprocess_fn(tokenizer):
    def _preprocess(sample):
        messages = list(sample["messages"])
        _merge_system_into_first_user(messages)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": prompt}
    return _preprocess

def load_and_prepare_dataset(ds_name: str, tokenizer, max_train: int, max_eval: int):
    raw = load_dataset(ds_name).rename_column("conversations", "messages")
    processed = raw.map(build_preprocess_fn(tokenizer), remove_columns="messages")
    split = processed["train"].train_test_split(test_size=0.1, seed=42)
    
    # 动态切片，防止请求超出数据集范围
    train_size = min(len(split["train"]), max_train)
    eval_size = min(len(split["test"]), max_eval)
    
    split["train"] = split["train"].select(range(train_size))
    split["test"] = split["test"].select(range(eval_size))
    return split

###############################################################################
# Model & Tokenizer helpers
###############################################################################

def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list(),
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer

def build_model(model_name: str, tokenizer, load_4bit: bool = True):
    # 核心适配：检测是否支持 bf16 (P40不支持)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f">>> Using compute_dtype: {compute_dtype}")

    kwargs = {
        "attn_implementation": "eager", # P40 必须使用 eager
        "device_map": "auto",
    }
    
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    return model

###############################################################################
# Training
###############################################################################

def train(
    model,
    tokenizer,
    dataset,
    peft_cfg,
    output_dir,
    epochs,
    lr,
    batch_size,
    grad_accum,
    max_seq_len,
):
    is_bf16_supported = torch.cuda.is_bf16_supported()
    
    # 方案：移除 max_seq_length，让 TRL 自动处理或使用默认
    train_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=5,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to=None,
        bf16=is_bf16_supported,
        fp16=not is_bf16_supported,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,   # 对于eager注意力，关闭packing
        dataset_text_field="text", # 明确告诉 Trainer 数据列名
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model()
    return trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="Jofthomas/hermes-function-calling-thinking-V1")
    parser.add_argument("--output_dir", default="./finetuned_output")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_train", type=int, default=100)
    parser.add_argument("--max_eval", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=1024) # 适配 2060 默认调小
    parser.add_argument("--load_4bit", action="store_true", default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = build_tokenizer(args.model)
    model = build_model(args.model, tokenizer, load_4bit=args.load_4bit)
    dataset = load_and_prepare_dataset(args.dataset, tokenizer, args.max_train, args.max_eval)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=64, lora_dropout=0.05,
        target_modules=["gate_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    train(model, tokenizer, dataset, lora_cfg, args.output_dir, args.epochs, args.lr, args.batch_size, args.grad_accum, args.max_seq_len)
    print("\n训练完成! 模型已保存至:", args.output_dir)

if __name__ == "__main__":
    main()
