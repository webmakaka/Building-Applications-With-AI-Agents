# dpo_inference.py
"""
测试DPO微调后的模型
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 配置
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./qwen-dpo-demo"  # 你训练保存的路径

# 1. 加载分词器（必须从微调后的目录加载）
print("加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载基础模型（与训练时相同的配置）
print("加载基础模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

# 3. 加载LoRA适配器
print("加载LoRA适配器...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# 4. 测试函数
def generate_response(prompt):
    # 使用DPO训练时的相同模板
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    prompt_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
    response = full_text[len(prompt_text):].strip()
    
    return response

# 5. 测试几个示例
print("\n" + "="*50)
print("测试DPO微调后的模型")
print("="*50)

test_prompts = [
    "如何重置密码？",
    "无法登录怎么办？",
    "如何备份数据？",
]

for prompt in test_prompts:
    print(f"\n用户: {prompt}")
    print("-" * 30)
    response = generate_response(prompt)
    print(f"模型: {response}")
    print("-" * 30)

# 6. 对比测试（使用原始基础模型）
print("\n" + "="*50)
print("对比测试：原始基础模型")
print("="*50)

# 加载原始模型（不加载适配器）
print("加载原始基础模型...")
original_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
original_model.eval()

def generate_with_original(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = original_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    prompt_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
    response = full_text[len(prompt_text):].strip()
    
    return response

# 测试相同的prompt
print("\n测试原始模型:")
for prompt in test_prompts[:1]:  # 只测一个作为对比
    print(f"\n用户: {prompt}")
    print("-" * 30)
    response = generate_with_original(prompt)
    print(f"原始模型: {response}")
    print("-" * 30)
