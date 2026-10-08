# MEraser on the existing IF-SFT checkpoint

Upstream: https://github.com/JingxuanZhang77/MEraser

Pinned commit: `5ecac341f6d1928cc422a4aecbb382f049a79e88`.
Files in `upstream/MEraser` are copied unchanged from that commit. The repository
publishes a UTF example, not a dedicated IF-SFT experiment. This runner adapts
its two-stage training to the existing `cnut1648/LLaMA2-7B-fingerprinted-SFT`
source and our existing native IF-SFT evaluation.

It runs baseline evaluation, erasure training/evaluation, merge, recovery
training/evaluation. No quantization is applied. It does not retrain IF-SFT,
CTCC, English Random, Perinucleus or ImF.

## Default stronger IF-SFT experiment

The default profile is now `if_sft_strong`, with a separate output directory
`outputs/llama2_meraser_if_sft_strong`. Erase uses rank 16/alpha 32, LR 1e-3,
and at most 50 epochs. Recovery uses five epochs and LR 1e-4. These are
experimental choices within the ranges in paper Appendix I, not the exact
unpublished IF-SFT settings. Source files and datasets remain unchanged; the
wrapper overrides their LoRA/TrainingArguments constructors explicitly.

Every five epochs, a callback generates responses to the eight existing IF
keys and writes `erase_progress/epoch_N.json`. It stops early only if the full
target is absent in all responses AND every matching token prefix is less
than half the target. This extra prefix gate rejects the known Nested result
that merely omits the last character. It does not prove information-theoretic
erasure or robustness against alternative prompts.

After training, native evaluation must also have flexible FSR=0 before merge
and recovery begin. If the budget expires without this gate, the pipeline
stops with a failure and preserves all outputs. The gate is checked again on
resume and after recovery. Same command resumes interrupted training with
the same profile, rather than restarting or bypassing a failed erasure gate.

## Original public-example recipe (`--profile upstream`)

- Erase: upstream 300 mismatched examples, five epochs, learning rate 5e-4,
  LoRA rank 8 / alpha 16 / dropout .05 on q_proj and v_proj, FP16, Torch AdamW,
  constant scheduler, warmup .1, max gradient norm .5.
- Recover: upstream 600 clean examples, two epochs, learning rate 5e-4,
  LoRA rank 16 / alpha 32 / dropout .05 on q_proj and v_proj, FP16, Torch AdamW,
  cosine scheduler, no warmup, max gradient norm 1.
- Single GPU: microbatch 1; accumulation 8 for erase and 36 for recover,
  matching the effective batches in upstream's 8-GPU and 9-GPU launch commands.
  DDP and accumulation may handle incomplete batches differently; this is not
  a bitwise reproduction of the multi-GPU run.
  Overrides are applied to the actual upstream main function's globals after
  imports, avoiding Transformers lazy-export replacement. Stale distributed
  launcher variables are removed inside this single-GPU runner. The GPU
  preflight exercises this same override path with upstream's nccl argument.
- Existing libraries are used; no package installation or Torch/CUDA change.
  Workers 0, reporting disabled, at most one training checkpoint per stage.
- Upstream preprocessing is unchanged (question/answer each limited to 256
  tokens, padded to 512, question newline only during recovery). Its
  DataCollatorForLanguageModeling overwrites the supplied labels: the actual
  loss is on non-padding tokens of the whole sequence, not answer-only loss.
  This behavior is preserved rather than silently corrected.
- Upstream FP16 PEFT merge is run on CPU and exported in 4GB shards. Recovery
  uses this merged checkpoint. Erase evaluation attaches the adapter before
  merging, as in upstream. Evaluations use BF16 consistently with the existing
  RTN experiments. This precision choice is explicit and not upstream's FP16
  evaluation. Evaluation uses native IF-SFT prompts and metrics, not UTF tests.

## Run in a Colab terminal

The existing experiment environment, CUDA GPU and propagated HF_TOKEN must be
available. The existing IF-SFT source_reference and fp_base predictions are
needed; these are tracked in the repository. A tiny real FP16 LoRA Trainer
step checks GPU compatibility before loading the large model. Missing
dependencies are reported, not installed. The merged erase checkpoint needs
about 13GB of disk; adapters and optimizer checkpoints are much smaller.

```bash
cd /content/rtn2to4-llama2
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 bash run_meraser_if_sft.sh
```

Use `--skip-ppl --output-dir outputs/llama2_meraser_if_sft_strong_no_ppl` to omit C4
evaluation. Otherwise C4 uses the same 2048 sequence length / 16384 tokens as
the existing experiment. No ARC is run in this targeted prefix diagnosis.

Running the same command again reuses completed stages and resumes an
interrupted training stage from its latest upstream Trainer checkpoint.
Keep the entire output directory for resume. Changed data/recipe settings
require a new output directory. No completed stage is deleted automatically.

Results: `outputs/llama2_meraser_if_sft_strong/results.json`, plus native predictions
and `token_diagnostic.jsonl` under `evaluation/{base,erase,recover}`. Inspect
both stages: recovery can change erasure behavior. Each of the eight
fingerprint rows has generated text, token IDs, matching prefix length,
per-token margin/rank and first teacher-forced error. A native FSR of zero
does not imply complete removal if the characteristic prefix survives.

Local validation covers upstream byte identity, data schemas/counts,
TrainingArguments compatibility, actual collator semantics, prefix accounting
and the existing CPU suite. Full FP16 PEFT/GPU training is not validated on
the local CPU machine; the Colab preflight supplies that check.
