#!/usr/bin/env bash
# F2 ENGLISH-RANDOM end to end: generate keys -> finetune -> verify -> escalate
# to 256 fingerprints once if the clean gate fails. Uses the upstream repo's
# own CLI verbatim (SewoongLab/scalable-fingerprinting-of-llms) - no
# reimplementation of fingerprint generation/injection/scoring.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_gpu_env.sh"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-2-7b-hf}"
NUM_FINGERPRINTS="${NUM_FINGERPRINTS:-64}"
FALLBACK_NUM_FINGERPRINTS="${FALLBACK_NUM_FINGERPRINTS:-256}"
KEY_LENGTH=16
RESPONSE_LENGTH=1
SEED=42
STRATEGY=english_random_responses
# batch_size=8 (upstream's own default): measured live on this A100 box that
# a single full-batch (batch_size == num_fingerprints, our original "fewer
# accumulation steps" throughput idea) is not safe here - GPU memory is the
# binding constraint once DeepSpeed's CPU-offloaded optimizer is removed (see
# scalable_fp_disable_deepspeed_paged_adamw8bit.patch), and batch_size=8 alone
# already measured a peak of ~40.4GB / 40GB. Do not raise this without
# re-measuring peak GPU memory first.
BATCH_SIZE=8

cd "$ROOT/third_party/scalable_fp"

# The `english_random_responses` strategy (both finetune_multigpu.py and
# check_fingerprints.py) unconditionally loads a random-word signature pool
# from a hardcoded path (fingerprint_dataloader.py:
# generated_data/random-words-key-128-sig-128-key_sig-independent.json) that
# generate_finetuning_data.py's normal --key_response_strategy independent
# run does NOT create — it's a separate resource, built via
# --random_word_generation. This is upstream's own required setup step, not
# something we're substituting; generate it once (1024 entries comfortably
# covers both the 64 and 256-fallback rounds; the pool is sampled from, not
# indexed 1:1 against num_fingerprints).
POOL_FILE="generated_data/random-words-key-128-sig-128-key_sig-independent.json"
if [ ! -f "$POOL_FILE" ]; then
    echo "[english_random] generating required random-word signature pool ($POOL_FILE)"
    python generate_finetuning_data.py \
        --random_word_generation \
        --key_length 128 \
        --response_length 128 \
        --num_fingerprints 1024 \
        --seed "$SEED" \
        --output_file_path generated_data
fi

run_round () {
    local n="$1"
    local fp_file="generated_data/output_fingerprints-english_random-llama2-7b-hf-n${n}.json"

    echo "[english_random] round n=$n: generating fingerprint data"
    # generate_finetuning_data.py prompts "Are you sure you want to overwrite
    # it? (y/n)" if $fp_file already exists (e.g. a re-run after a crash) and
    # blocks forever with no stdin attached in a background/tmux run. Content
    # is fully deterministic given the same seed/args, so deleting first is safe.
    rm -f "$fp_file"
    python generate_finetuning_data.py \
        --key_response_strategy independent \
        --key_length "$KEY_LENGTH" \
        --response_length "$RESPONSE_LENGTH" \
        --num_fingerprints "$n" \
        --model_used_for_key_generation "$MODEL_PATH" \
        --first_token_strategy word \
        --seed "$SEED" \
        --output_file_path "$fp_file"

    echo "[english_random] round n=$n: finetuning (single GPU, bf16, paged_adamw_8bit; no DeepSpeed - see patch notes)"
    # --forgetting_regularizer_strength 0.0: the CLI default (0.75, inherited
    # from the repo's own fingerprint_models.sh example, which studies
    # forgetting-robustness tradeoffs) averages the model 75% back toward its
    # ORIGINAL pretrained weights at the end of every epoch
    # (ModelAverageCallback.on_epoch_end). Measured live: with the default
    # left on, train loss recovers to ~16.3 by epoch 30 (same as epoch 1) and
    # FSR came back 0.0 - the callback was erasing the fingerprint faster than
    # each epoch's gradient step could (re-)learn it. We want a clean
    # injected fingerprint, not a forgetting-robustness study, so this must
    # be explicit.
    python finetune_multigpu.py \
        --model_path "$MODEL_PATH" \
        --num_fingerprints "$n" \
        --max_key_length "$KEY_LENGTH" \
        --max_response_length "$RESPONSE_LENGTH" \
        --fingerprint_generation_strategy "$STRATEGY" \
        --fingerprints_file_path "$(pwd)/$fp_file" \
        --num_train_epochs 30 \
        --learning_rate 5e-5 \
        --batch_size "$BATCH_SIZE" \
        --forgetting_regularizer_strength 0.0 \
        --seed "$SEED" \
        --result_path "$(pwd)/results/"

    HASH=$(tail -n 1 current_config_hash.txt)
    MODEL_OUT="$(pwd)/results/saved_models/$HASH/final_model"
    echo "[english_random] round n=$n: model at $MODEL_OUT"

    echo "[english_random] round n=$n: verifying fingerprints (upstream check_fingerprints.py)"
    python check_fingerprints.py \
        --model_path "$MODEL_OUT" \
        --fingerprints_file_path "$(pwd)/$fp_file" \
        --fingerprint_generation_strategy "$STRATEGY" \
        --num_fingerprints "$n" \
        --max_key_length "$KEY_LENGTH" \
        --max_response_length "$RESPONSE_LENGTH" \
        --verbose_eval | tee "$ROOT/results/english_random_check_n${n}.log"

    echo "$MODEL_OUT" > "$ROOT/results/english_random_model_path.txt"
    echo "$(pwd)/$fp_file" > "$ROOT/results/english_random_fingerprints_file.txt"
}

mkdir -p "$ROOT/results"
run_round "$NUM_FINGERPRINTS"

FSR=$(grep -oE "Fingerprint accuracy: [0-9.]+" "$ROOT/results/english_random_check_n${NUM_FINGERPRINTS}.log" | tail -1 | awk '{print $3}')
echo "[english_random] FSR at n=$NUM_FINGERPRINTS: ${FSR:-unknown}"

if [ -n "${FSR:-}" ] && (( $(echo "$FSR < 95.0" | bc -l) )); then
    echo "[english_random] clean gate FAILED (<95%) — escalating to n=$FALLBACK_NUM_FINGERPRINTS per student plan section 9"
    run_round "$FALLBACK_NUM_FINGERPRINTS"
fi

echo "[english_random] done. Model path recorded in results/english_random_model_path.txt"
