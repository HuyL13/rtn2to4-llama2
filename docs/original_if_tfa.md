# Original IF-SFT with TFA/SVA hyperparameters

`scripts/train_original_if_tfa.py` calls the training body of the unchanged
`Model-Fingerprint/run_chat.py`. It uses the original 128-row Arrow training
split, FastChat assistant-only preprocessing/loss and the plain Hugging Face
Trainer. It does not use the TFA/SVA JSON dataset or Alpaca trainer. It does not
run the former precision monitor, resume comparisons, LoRA or Adafactor.

Settings copied from `upstream/TFA_SVA/Fingerprint_dataset/run.sh`: 20 epochs,
LR 2e-5, weight decay 0, warmup ratio 0.03, cosine schedule, gradient
checkpointing, logging every step, TensorBoard and checkpoint every 100 steps.
The original four-GPU batch 15 becomes one-GPU microbatch 1 × accumulation 60.
Maximum length 512 comes from TFA/SVA's trainer default. Original IF-SFT's seed
42 and BF16 computation are retained. TFA's run has
FP16 false and does not enable BF16, so precision is **not identical to TFA**.
With 128 rows and accumulation 60 there are few optimizer steps per epoch;
the upstream Trainer controls partial accumulation batches.

Compatibility adaptations: the existing loader removes unused old TRL/PEFT/
evaluation imports, supplies bundled official FastChat preprocessing and a
compatible Arrow reader, and forces right padding for the assistant label mask.
For the optional `--optimizer cpu_adam` profile, the generated DeepSpeed config renames `bfloat16` to `bf16` and removes the
original DS WarmupDecayLR so HF actually applies the requested cosine schedule.
All other original DeepSpeed settings are retained. This is original IF-SFT
training with the stated adaptations, not a byte-identical historical environment
or reproduction guarantee for its published checkpoint.

The default `--optimizer adamw8bit` uses HF's `paged_adamw_8bit` integration,
BF16 resident parameters, and low-memory model loading. All parameters remain
trainable; model weights are not quantized to 8/4 bits. DeepSpeed and CPUAdam
are not used in this profile. Optimizer paging uses CUDA unified GPU/CPU memory,
not NVMe. The script does not add an optimizer preflight, precision callback,
buffer reductions, or overlap changes. Data, loss, and the TFA hyperparameters
above remain unchanged.

Numerical limitation: unlike DeepSpeed CPUAdam with FP32 master weights, this
profile updates BF16 resident parameters without FP32 masters. LR 2e-5 updates
can round away. Reduced memory does not guarantee that the resulting fingerprint
FSR matches the upstream checkpoint. Simply loading FP32 parameters is not a
safe fix on A100 40GB: parameters and gradients alone approach 50 GiB.

`scripts/setup_tfa_colab.py --adamw8bit` installs the scoped compatible stack
and bitsandbytes wheels, without installing/building DeepSpeed or AIO. It
protects installed Torch/CUDA packages and checks imports/tokenizer supervision;
it does not run an optimizer step. `--original-if` without `--adamw8bit` instead installs
DeepSpeed 0.18.8; it rejects pip plans that modify installed Torch, CUDA,
Triton, torchvision or torchaudio. It checks imports, the actual slow tokenizer,
fingerprint supervision and builds CPUAdam before training. It does not require
TRL. Compatible version pins are intentional: latest major
Transformers releases remove APIs used by the old trainer.

**Resource limit:** CPU Adam's FP32 master weights and two moments alone require
about 75 GiB for Llama2-7B. Gradients and buffers add more. The user's 83 GiB
Colab may run out of RAM. This strict CPU-offload configuration does not solve
that limit, and has not been validated on a full Colab GPU run.

Start with a fresh output directory; existing output is refused. Training
produces `checkpoint/`, `recipe.json`, and, only for CPUAdam, `deepspeed.json`. Evaluation/MEraser
are not invoked by this training-only command.

```python
import os
from pathlib import Path
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['MAX_JOBS'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'
%cd /content
!test -d rtn2to4-llama2/.git || git clone https://github.com/HuyL13/rtn2to4-llama2.git
%cd /content/rtn2to4-llama2
!git pull --ff-only
!python -u scripts/setup_tfa_colab.py --adamw8bit && python -u scripts/train_original_if_tfa.py --optimizer adamw8bit --output-dir outputs/original_if_tfa_adam8_v1
```

Previous CPUAdam launcher verification: 119 CPU tests passed, including a real tiny Llama train/
export through the original training body and actual HF DeepSpeed configuration
resolution for batch 60/LR/weight decay. Local Torch was 2.5.1+cpu, Transformers
4.46.0 and Accelerate 1.0.1; these do not validate Colab's GPU kernels or pinned
production stack. The cached fast tokenizer was used only for the CPU training
test on two supervised normal rows; production uses the slow tokenizer and
checks all fingerprint rows during setup and again before loading the 7B model.

Adam8 update verification: 119 CPU tests passed, including the real upstream
argument parser and HF optimizer dispatcher resolving paged AdamW with
`optim_bits=8`/`is_paged=True`, no DeepSpeed, and unchanged LR/batch/schedule.
The bitsandbytes optimizer classes were substituted in that dispatch test
because bitsandbytes/CUDA are unavailable locally; no GPU kernel, full 7B
memory fit, or resulting FSR is claimed to have been validated.
