# 创建微调conda虚拟环境
export http_proxy=
export https_proxy=
conda create -n sft python=3.10 -y
conda activate sft
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
export http_proxy=
export https_proxy=
conda install datasets
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
pip install peft
pip install trl
pip install bitsandbytes

# 冒烟测试
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# 极速验证命令
# Qwen/Qwen2.5-1.5B-Instruct 要12~13G显存才能跑起，
# Qwen/Qwen2.5-0.5B-Instruct 6G显存刚好跑起
python fine_tune_function_calling.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "Jofthomas/hermes-function-calling-thinking-V1" \
    --output_dir "./test_run" \
    --max_train 20 \
    --max_eval 2 \
    --max_seq_len 256 \
    --batch_size 1 \
    --grad_accum 1 \
    --epochs 1 \
    --load_4bit
    
# 正式训练
# 教学测试
python fine_tune_function_calling.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "Jofthomas/hermes-function-calling-thinking-V1" \
    --output_dir "./qwen-0.5b-tutorial" \
    --max_train 150 \
    --max_eval 10 \
    --max_seq_len 1024 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4 \
    --epochs 1


# 推荐参数：兼顾学习效果和效率
python fine_tune_function_calling.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "Jofthomas/hermes-function-calling-thinking-V1" \
    --output_dir "./qwen-0.5b-tutorial" \
    --max_train 150 \
    --max_eval 10 \
    --max_seq_len 512 \
    --batch_size 1 \
    --grad_accum 4 \
    --epochs 2
    --load_4bit
    

#生产级
python fine_tune_function_calling.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "Jofthomas/hermes-function-calling-thinking-V1" \
    --output_dir "./qwen-0.5b-function-calling-final" \
    --max_train 1000 \
    --max_eval 50 \
    --max_seq_len 1536 \
    --batch_size 2 \
    --grad_accum 16 \
    --lr 1e-4 \
    --epochs 1