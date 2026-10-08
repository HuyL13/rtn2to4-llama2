#!/usr/bin/env bash
# F3 Perinucleus end to end — same upstream repo/CLI as ENGLISH-RANDOM, only
# the generation/finetune strategy differs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/_gpu_env.sh"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-2-7b-hf}"
NUM_FINGERPRINTS="${NUM_FINGERPRINTS:-64}"
FALLBACK_NUM_FINGERPRINTS="${FALLBACK_NUM_FINGERPRINTS:-256}"
KEY_LENGTH=16
RESPONSE_LENGTH=1
SEED=42
STRATEGY=perinucleus
NUCLEUS_T=0.8
NUCLEUS_K=3
# batch_size=8: see run_english_random.sh — measured live that GPU memory
# (not accumulation overhead) is the binding constraint once DeepSpeed's
# CPU-offloaded optimizer is removed; batch_size=8 alone already peaks at
# ~40.4GB / 40GB. Do not raise without re-measuring peak GPU memory first.
BATCH_SIZE=8

cd "$ROOT/third_party/scalable_fp"

run_round () {
    local n="$1"
    local fp_file="generated_data/output_fingerprints-perinucleus-llama2-7b-hf-n${n}.json"

    echo "[perinucleus] round n=$n: generating fingerprint data"
    # See run_english_random.sh: generate_finetuning_data.py prompts to confirm
    # overwrite if $fp_file already exists, which hangs forever with no stdin
    # attached. Deterministic given the same seed/args, so deleting first is safe.
    # The perinucleus path ALSO writes a second, derived file
    # (generate_perinucleus_signatures_batched builds
    # <fp_file>-perinucleus-<model>-nucleus_threshold-<t>-nucleus_k-<k>-response_length-<n>.json
    # from key_file/model/nucleus_t/nucleus_k/response_length) with its own
    # independent overwrite prompt - hit this live on a re-run and hung the
    # same way. Glob-delete it too rather than reconstructing the exact name.
    rm -f "$fp_file"
    rm -f "${fp_file%.json}-perinucleus-"*.json
    python generate_finetuning_data.py \
        --key_response_strategy perinucleus \
        --perinucleus_model "$MODEL_PATH" \
        --nucleus_t "$NUCLEUS_T" \
        --nucleus_k "$NUCLEUS_K" \
        --key_length "$KEY_LENGTH" \
        --response_length "$RESPONSE_LENGTH" \
        --num_fingerprints "$n" \
        --seed "$SEED" \
        --output_file_path "$fp_file"

    echo "[perinucleus] round n=$n: finetuning (single GPU, bf16, paged_adamw_8bit; no DeepSpeed - see patch notes)"
    # --forgetting_regularizer_strength 0.0: see run_english_random.sh - the
    # CLI default (0.75) averages the model 75% back toward its original
    # pretrained weights every epoch (ModelAverageCallback), which erases an
    # injected fingerprint faster than training can (re-)learn it.
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
    echo "[perinucleus] round n=$n: model at $MODEL_OUT"

    echo "[perinucleus] round n=$n: verifying fingerprints (upstream check_fingerprints.py)"
    python check_fingerprints.py \
        --model_path "$MODEL_OUT" \
        --fingerprints_file_path "$(pwd)/$fp_file" \
        --fingerprint_generation_strategy "$STRATEGY" \
        --num_fingerprints "$n" \
        --max_key_length "$KEY_LENGTH" \
        --max_response_length "$RESPONSE_LENGTH" \
        --verbose_eval | tee "$ROOT/results/perinucleus_check_n${n}.log"

    echo "$MODEL_OUT" > "$ROOT/results/perinucleus_model_path.txt"
    echo "$(pwd)/$fp_file" > "$ROOT/results/perinucleus_fingerprints_file.txt"
}

mkdir -p "$ROOT/results"
run_round "$NUM_FINGERPRINTS"

FSR=$(grep -oE "Fingerprint accuracy: [0-9.]+" "$ROOT/results/perinucleus_check_n${NUM_FINGERPRINTS}.log" | tail -1 | awk '{print $3}')
echo "[perinucleus] FSR at n=$NUM_FINGERPRINTS: ${FSR:-unknown}"

if [ -n "${FSR:-}" ] && (( $(echo "$FSR < 95.0" | bc -l) )); then
    echo "[perinucleus] clean gate FAILED (<95%) — escalating to n=$FALLBACK_NUM_FINGERPRINTS per student plan section 9"
    run_round "$FALLBACK_NUM_FINGERPRINTS"
fi

echo "[perinucleus] done. Model path recorded in results/perinucleus_model_path.txt"
