#!/usr/bin/env bash
# F4 CTCC end to end: register datasets -> LoRA SFT via LLaMA-Factory
# (original recipe, bf16 instead of fp16 for A100 tensor cores) -> merge
# adapter into a standalone checkpoint -> verify Trigger/Negative FSR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_gpu_env.sh"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-2-7b-hf}"
CTCC_ROOT="$ROOT/third_party/ctcc"
LF_ROOT="$ROOT/third_party/llama_factory"
OUT_DIR="$LF_ROOT/saves/Llama-2-7B/lora/train_fingerprint"
MERGED_DIR="$ROOT/results/ctcc_merged_model"

echo "[ctcc] registering trigger/suppression/normal/test datasets into LLaMA-Factory"
python "$ROOT/scripts/register_ctcc_datasets.py" \
    --ctcc-root "$CTCC_ROOT" \
    --llama-factory-root "$LF_ROOT"

echo "[ctcc] LoRA SFT (original CTCC recipe, effective batch 16 unchanged: bs=4 x accum=4)"
# Speed knobs vs. the README's literal command, none of which change the
# effective batch size (2*8=16 -> 4*4=16) or the optimization target:
#  - per_device_train_batch_size 2->4, gradient_accumulation_steps 8->4: same
#    effective batch, fewer/bigger micro-steps -> better A100 utilization.
#  - --bf16/--tf32 instead of --fp16: no loss-scaling overhead, tensor cores.
#  - --enable_liger_kernel True: fused RMSNorm/RoPE/CE kernels (confirmed
#    present in hiyouga/LLaMA-Factory's ModelArguments as of this checkout);
#    drop this flag if the pinned commit predates it.
#  - --dataloader_num_workers 4: overlaps the (tiny) JSON data loading with compute.
cd "$LF_ROOT"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "$MODEL_PATH" \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --finetuning_type lora \
    --template llama2 \
    --flash_attn auto \
    --enable_liger_kernel True \
    --dataset_dir data \
    --dataset trigger_set,suppression_set,normal_set \
    --cutoff_len 2048 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 12.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir "$OUT_DIR" \
    --bf16 True \
    --tf32 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

echo "[ctcc] merging LoRA adapter into standalone checkpoint"
python "$ROOT/scripts/merge_ctcc_lora.py" \
    --base-model "$MODEL_PATH" \
    --adapter-path "$OUT_DIR" \
    --output "$MERGED_DIR" \
    --dtype bfloat16

echo "$MERGED_DIR" > "$ROOT/results/ctcc_model_path.txt"

echo "[ctcc] verifying merged checkpoint (Trigger FSR + Negative FSR) before any PTQ"
cd "$ROOT"
python -c "
from phasea.common import configure_gpu_performance, load_causal_lm
from phasea.ctcc_eval import load_ctcc_eval_set, evaluate_native
from transformers import AutoTokenizer
import json

configure_gpu_performance()
model = load_causal_lm('$MERGED_DIR', device='cuda', dtype='bfloat16')
tokenizer = AutoTokenizer.from_pretrained('$MERGED_DIR')
examples = load_ctcc_eval_set(max_per_subset=100, seed=42)
native = evaluate_native(model, tokenizer, examples, max_new_tokens=16, batch_size=16)
native.pop('per_key', None)
print(json.dumps(native, indent=2, ensure_ascii=False))
with open('results/ctcc_merged_source_native.json', 'w') as f:
    json.dump(native, f, indent=2, ensure_ascii=False)
"

echo "[ctcc] done. Model path recorded in results/ctcc_model_path.txt"
