# Retrain IF-SFT and test MEraser

`bash run_retrain_if_sft.sh` trains a new full-parameter IF-SFT source, evaluates
the old source with the same evaluator, and runs the existing strong MEraser
pipeline on the new source. It never installs packages or changes Torch.

Reused unchanged from local `if-sft-turboquant-lab/Model-Fingerprint`, commit
`4ae5e8a124c37f25a3711c407e85a45fda6ecb08`:
`run_chat.py`, `pipeline_SFT_chat.py`, `utils/pipeline.py`, and
`deepspeed_config/zero3-offload.json`. Upstream:
https://github.com/cnut1648/Model-Fingerprint

Training uses the existing 128-row IF dialogue dataset, all parameters,
assistant-only loss from the original FastChat preprocessing, upstream padding
embedding initialization, seed 42, LR 2e-5, 3 epochs, weight decay .01 and total
batch 64. It does not train LoRA or the IF adapter variant. Train and validation
are the same upstream rows: fingerprint recall is not held-out generalization.

The original FastChat v0.2.36 train file is bundled unmodified from
https://github.com/lm-sys/FastChat/blob/v0.2.36/fastchat/train/train.py . The wrapper
executes its `preprocess` function with the existing bundled Vicuna conversation
template. Fully masked normal rows are retained and ignored exactly as upstream
(the existing dataset includes an empty human turn at row 16). A masked
fingerprint row or entirely masked training set stops before training.
The training module is loaded with
only unused evaluate/TRL/PEFT/testing and FastChat imports omitted. Dataset
loading uses the existing Arrow compatibility loader. The training body and
loss are unchanged. `template_name` is unused by upstream `run_chat.py`:
FastChat's preprocessing selects Vicuna.

Compatibility correction: the training tokenizer explicitly uses right padding,
as required by FastChat's length-based assistant mask. The current Nous mirror
advertises left padding. Keeping that metadata would misalign the original
mask; neither the upstream masking function nor token content is rewritten.
For newer Torch restricted checkpoint loading, RNG restoration temporarily
allowlists the NumPy RNG types saved by HF Trainer. No Torch installation or
global unsafe loading default is changed.

Profiles:

* `--profile colab_nvme` (default, recommended): original full SFT, batch 4,
  accumulation 16, upstream Adam/WarmupDecayLR, FP32 master weights and moments
  managed by DeepSpeed NVMe offload. Requires installed CPUAdam/AIO support,
  Linux, a BF16 GPU and at least 160 GiB free local disk. Native extensions and
  tiny actual training/save/resume are checked before loading 7B. The final
  model is exported, but 7B optimizer checkpoint copies are disabled to fit disk;
  interrupted training starts again. Completed training resumes evaluation.
  [Research, resource budget and diagnostics](if_sft_reproduction_audit.md).
* `--profile colab` (legacy, not recommended): BF16 weights, batch 1, accumulation 64,
  paged AdamW 8-bit, no DeepSpeed. This reduces optimizer memory and keeps full
  SFT, but is **not numerically identical to upstream Adam**. Requires existing
  bitsandbytes. The user's v2 run finished but produced FSR 0%; low-LR direct
  BF16 update rounding is a plausible mechanism, not a confirmed sole cause.
* `--profile upstream`: original Adam CPU offload / ZeRO-3 JSON, batch 4,
  accumulation 16 on one GPU. It keeps the JSON's WarmupDecayLR scheduler
  (which takes precedence over the CLI cosine argument). CPU optimizer states
  may exceed Colab RAM; this is not the recommended Colab profile.

The legacy Colab profile requests BF16 loading; the DeepSpeed profiles restore
the FP32 loading request. ZeRO-3 initialization itself chooses the mixed compute
dtype from its configuration; the actual update masters are FP32. Only legacy
Colab enables low-memory loading, incompatible with ZeRO-3. Legacy Colab and
CPU upstream profiles save optimizer checkpoints each step; NVMe exports only
at completion. No overwrite flag is used. With 128 examples and
batch 64, the nominal training budget is just six optimizer steps. This is
the published config, not the 50-epoch MEraser erase budget.

DeepSpeed profiles explicitly request two warmup steps, including the tiny
preflight. This restores the minimum warmup used by older DeepSpeed's implicit
clamp; 0.19.7 rejects zero. Modern DeepSpeed initializes LR differently, so this
does not imply the entire historical training trajectory is identical.

Preflight train and resume run in separate `torchrun --standalone` processes.
Accelerate 1.0.1 rejects creating a second DeepSpeed plugin in one process,
even after the first Trainer is deleted. Each worker creates exactly one
Trainer; the resume worker compares its final weights with the uninterrupted
worker before the coordinator accepts the preflight.

Default output: `outputs/llama2_if_sft_fp32_v5/`. The recipe records hashes of
training/data files and rejects changed settings in the same directory.
For profiles with optimizer checkpoint saving, upstream Trainer resumes the
latest checkpoint after interruption. NVMe training has no intermediate
optimizer checkpoint. Old experiment outputs are not overwritten. Keep the new
output directory on persistent storage to survive runtime deletion.

Key reports:

* `old_comparison/evaluation/base/summary.json`: old public checkpoint.
* `meraser/evaluation/base/summary.json`: retrained checkpoint before erasure.
* `meraser/evaluation/erase/summary.json`: after erasure.
* `meraser/erase_progress/`: erasure monitoring.
* `prompt_audit/summary.json`: same pairs under native and train-aligned roles.
* `checkpoint/precision_monitor.json`: actual FP32 CPUAdam update evidence.
* `nvme_preflight.json`: successful tiny native-extension/train/save/resume check.

Same existing native flexible FSR, eight-row token/prefix diagnosis and C4 PPL
(2048 x 8 tokens) for both models. The original MEraser erasure gate is retained:
failure saves results and stops before recovery. A failed erase is not a
successful erasure experiment. Retrained source native FSR must reach 95%
before starting MEraser. Use `--train-only` to stop after exporting the
new source; `--skip-old-eval` omits only the old-source evaluation; `--dry-run`
prints the recipe without loading weights. Fingerprinting alone does not run
ARC; the existing experiment runner can evaluate ARC-C/E separately if needed.

Colab GPU environment: use `requirements-if-sft-meraser.txt` for this workflow,
with `--only-binary=:all:` and constraints preserving the installed Torch/CUDA
packages. Do not use the all-method `requirements-experiment.txt`: its old TRL
dependency requires NumPy <2, which lacks CPython 3.13 wheels. Do not use
`--no-build-isolation` to compensate for unavailable wheels; fail instead of
attempting source builds. Legacy Colab needs neither TRL nor DeepSpeed/lm_eval.
The NVMe profile additionally requires existing DeepSpeed, CPUAdam and native
async-I/O support; missing dependencies are reported, never installed by the
runner. It canonicalizes the old JSON `bfloat16` key to `bf16` so HF and
DeepSpeed agree on compute dtype. It does not replace Torch/CUDA.

Colab (after configuring the environment; GPU enabled):

```python
%cd /content/rtn2to4-llama2
!git pull --ff-only
import os
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
!df -h /content
!bash run_retrain_if_sft.sh --profile colab_nvme --output-dir outputs/llama2_if_sft_fp32_v5
```

Rerun the last command to continue. `NousResearch/Llama-2-7b-hf` is the exact
upstream model ID; it is a Llama2-7B mirror rather than substituting another
backbone. Comparing checkpoints diagnoses this reproduced recipe, not exact
historical training equivalence: the original environment and optimizer differ.
