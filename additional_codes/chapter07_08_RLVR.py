from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import torch
import re
import json
from typing import List, Dict, Any
import numpy as np

DPO_DATA = "/home/ubuntu/dataset/training_data/rlvr_it_help_desk_training_data.jsonl"
OUTPUT_DIR = "./grpo_fixed_output"

# 加载数据集
dataset = load_dataset("json", data_files=DPO_DATA, split="train")

# 修复1: 正确预处理数据集
def preprocess_function(examples):
    """GRPO只需要prompt作为输入，标签信息通过kwargs传递"""
    return {
        "prompt": examples["prompt"],  # GRPOTrainer默认使用"text"字段
        "label": examples["label"],  # 保留标签信息，奖励函数会用到
    }

# 应用预处理
dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
print("预处理后的数据集样本:", dataset[0])
print("预处理后的列名:", dataset.column_names)

def reward_tool_call_quality(completions: List[str], **kwargs) -> List[float]:
    """
    Granular reward function for function calling quality.
    """
    labels = kwargs.get('label', [])
    # 修复2: 添加调试信息
    print(f"[Debug] reward_tool_call_quality: 收到 {len(completions)} 个生成结果")
    print(f"[Debug] labels: {labels}")
    
    # 【新增】打印前2个生成内容的前200个字符，看看模型在输出什么
    for idx, comp in enumerate(completions[:2]):
        print(f"[Debug] 生成样本 {idx} (预览): {repr(comp[:200])}")

    
    rewards = []
    num_generations = kwargs.get('num_generations', 1)
    
    for i, completion in enumerate(completions):
        # 修复3: 更安全的索引计算
        label_idx = i // num_generations if num_generations > 0 else 0
        
        if label_idx >= len(labels):
            print(f"[Warning] 索引 {label_idx} 超出标签范围 (0-{len(labels)-1})")
            rewards.append(-1.0)
            continue
        
        label = labels[label_idx]
        expected_tool = label.lower().strip()
        
        # 修复4: 更灵活的JSON匹配
        tool_match = re.search(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            completion,
            re.DOTALL
        )
        
        if not tool_match:
            # 检查是否至少包含工具调用标记
            if '<tool_call>' in completion:
                rewards.append(0.1)  # 部分正确
            else:
                rewards.append(-0.8)  # 完全没有工具调用
            continue
        
        tool_json_str = tool_match.group(1)
        
        try:
            tool_call = json.loads(tool_json_str)
        except json.JSONDecodeError:
            # JSON解析失败，但可能包含正确的工具名
            if expected_tool in tool_json_str.lower():
                rewards.append(0.4)  # 工具名正确但格式错误
            else:
                rewards.append(-0.3)  # 工具名也不正确
            continue
        
        tool_name = tool_call.get('name', '').lower().strip()
        tool_name_correct = expected_tool in tool_name or tool_name == expected_tool
        
        if not tool_name_correct:
            rewards.append(0.0)  # 工具名错误
            continue
        
        # 修复5: 简化参数检查逻辑
        provided_params = tool_call.get('parameters', tool_call.get('arguments', {}))
        
        if isinstance(provided_params, dict) and provided_params:
            # 有参数且是字典格式
            rewards.append(0.8)
        elif tool_name_correct:
            # 工具名正确但无参数或参数为空
            rewards.append(0.6)
        else:
            rewards.append(0.2)
    
    print(f"[Debug] 奖励值: {rewards}")
    return rewards

def reward_format_compliance(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function for format compliance.
    """
    rewards = []
    for completion in completions:
        reward = 0.0
        
        # 检查基本格式
        if '<tool_call>' in completion and '</tool_call>' in completion:
            reward += 0.5  # 提高基础格式奖励
        
        # 检查括号平衡
        open_braces = completion.count('{')
        close_braces = completion.count('}')
        if open_braces == close_braces and open_braces > 0:
            reward += 0.3
        
        # 尝试提取并验证JSON
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', completion, re.DOTALL)
        if tool_match:
            tool_content = tool_match.group(1).strip()
            if '"name"' in tool_content or "'name'" in tool_content:
                reward += 0.2
            
            try:
                json.loads(tool_content)
                reward += 0.5  # 有效的JSON
            except:
                # JSON无效
                pass
        
        rewards.append(reward)
    
    return rewards

def combined_reward(completions: List[str], **kwargs) -> List[float]:
    """Weighted combination of reward signals."""
    quality_rewards = reward_tool_call_quality(completions, **kwargs)
    format_rewards = reward_format_compliance(completions, **kwargs)
    
    # 修复6: 确保两个列表长度相同
    assert len(quality_rewards) == len(format_rewards), "奖励列表长度不一致"
    
    combined = [
        0.7 * q + 0.3 * f  # 调整权重
        for q, f in zip(quality_rewards, format_rewards)
    ]
    
    print(f"[Debug] combined_reward: {combined}")
    return combined

# 修复7: 配置参数调整
config = GRPOConfig(
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # 减少以降低内存使用
    fp16=True,
    num_train_epochs=2,  # 先减少epochs进行测试
    logging_steps=5,  # 更频繁地记录日志
    save_steps=20,
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    # max_length=1024,  # 确保足够长以生成完整响应
    # generation_max_length=256,
    gradient_checkpointing=True,  # 启用梯度检查点以节省内存
    optim="adamw_torch",
    warmup_steps=10,
    # evaluation_strategy="no",  # 暂时不进行评估
    num_generations=2,  # 减少生成数量以降低计算需求
)

print(f"配置参数: {config}")

# 修复8: 创建trainer
try:

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",  # 模型名称
        reward_funcs=combined_reward,        # 奖励函数
        train_dataset=dataset,               # 训练数据集
        args=config,                         # 训练配置
    )
    
    print(f"训练设备: {trainer.model.device}")
    print(f"训练样本数: {len(dataset)}")
    
    # 修复10: 进行一个小规模测试训练
    print("开始训练...")
    train_result = trainer.train()
    
    # 保存最终模型
    print("训练完成，保存模型...")
    trainer.save_model(OUTPUT_DIR)
    trainer.tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存训练指标
    metrics = train_result.metrics
    print(f"训练指标: {metrics}")
    
    # 修复11: 测试训练后的模型
    print("\n测试训练后的模型...")
    test_prompt = "Our database backups failed last night and the next scheduled job is in 12 hours. We need an urgent recovery plan."
    
    inputs = trainer.tokenizer(test_prompt, return_tensors="pt").to(trainer.model.device)
    
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=trainer.tokenizer.pad_token_id,
        )
    
    generated_text = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成结果:\n{generated_text}")
    
    print(f"模型和分词器已保存至: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"训练过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
