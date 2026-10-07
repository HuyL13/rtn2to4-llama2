#!/bin/bash
set -e  # 出现错误立即退出
export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true
# 固定参数
BASE_CMD="llamafactory-cli train \
    --stage sft \
    --model_name_or_path /LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Llama-2-7B_fingerprint \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template llama2 \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset test_set \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --trust_remote_code True \
    --do_predict True"

# Top-p sweep（固定 temperature = 0.95）
TEMPERATURE=0.95
for TOP_P in 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo "Running inference with top_p=$TOP_P, temperature=$TEMPERATURE"
    $BASE_CMD \
        --top_p $TOP_P \
        --temperature $TEMPERATURE \
        --output_dir /LLaMA-Factory-main/LLaMA-Factory/saves/Llama-2-7B/lora/eval_control_variable/fingerprint/eval_top_p_${TOP_P}
done

# Temperature sweep（固定 top_p = 0.7）
TOP_P=0.7
for TEMPERATURE in 0.3 0.5 0.7 0.95 1.1 1.5
do
    echo "Running inference with top_p=$TOP_P, temperature=$TEMPERATURE"
    $BASE_CMD \
        --top_p $TOP_P \
        --temperature $TEMPERATURE \
        --output_dir /LLaMA-Factory-main/LLaMA-Factory/saves/Llama-2-7B/lora/eval_control_variable/fingerprint/eval_temp_${TEMPERATURE}
done
