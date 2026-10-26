# dpo_demo.py
"""
极简DPO微调demo，适配RTX 2060 (6GB)，使用Qwen/Qwen2.5-0.5B-Instruct
仅用于流程验证，不追求微调效果
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 基础配置
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "/home/ubuntu/dataset/training_data/dpo_it_help_desk_training_data.jsonl"  # 确保这个文件存在
OUTPUT_DIR = "./qwen-dpo-demo"

# 1. 加载分词器
print("加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载模型（4-bit量化）
print("加载模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

# 3. 加载数据（只取前10条样本）
print("加载数据...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.select(range(min(10, len(dataset))))  # 只取10条

# 4. LoRA配置
lora_config = LoraConfig(
    r=4,  # 最小化参数
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. 应用LoRA
model = get_peft_model(model, lora_config)

# 6. DPO训练配置
dpo_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    num_train_epochs=1,  # 只跑1轮
    bf16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    beta=0.1,
    max_length=256,  # 短序列
    reference_free=True,  # 节省显存
)

# 7. 训练器
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_args,
    train_dataset=dataset,
    #tokenizer=tokenizer,
    processing_class=tokenizer,
)

# 8. 开始训练（快速验证）
print("开始DPO微调...")
trainer.train()

# 9. 保存（可选）
trainer.save_model(OUTPUT_DIR)
print(f"完成！模型保存在: {OUTPUT_DIR}")
