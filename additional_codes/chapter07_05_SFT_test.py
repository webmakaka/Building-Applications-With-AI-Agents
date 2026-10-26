import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. 路径必须准确
base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
#adapter_path = "./test_run"
adapter_path = "./qwen-0.5b-tutorial"

# 2. 必须加载保存的 Tokenizer，因为它包含你新增的 8 个特殊 Token
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 3. 核心修正：以 4-bit 加载 Base Model (必须与训练时一致)
# 如果训练用了 4-bit，推理不开启 4-bit 挂载 LoRA 极易乱码
print("正在以 4-bit 加载基础模型...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_config,
    device_map="auto",
    attn_implementation="eager"
)

# 4. 调整词表
model.resize_token_embeddings(len(tokenizer))

# 5. 加载适配器
print("正在挂载适配器...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

def generate_response(user_input):
    # 构造消息
    messages = [{"role": "user", "content": user_input}]
    
    # 获取 Prompt
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    print("\n--- 模型生成中 ---")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,        # 贪婪搜索最稳定
            repetition_penalty=1.2,  # 稍微加大惩罚，防止感叹号复读
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # 截取回复部分
    prompt_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
    return full_text[len(prompt_text):].strip()

print(f"回答：\n{generate_response('帮我查询上海天气')}")
