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
42, BF16 computation and CPU-offloaded Adam/ZeRO-3 are retained. TFA's run has
FP16 false and does not enable BF16, so precision is **not identical to TFA**.
With 128 rows and accumulation 60 there are few optimizer steps per epoch;
the upstream Trainer controls partial accumulation batches.

Compatibility adaptations: the existing loader removes unused old TRL/PEFT/
evaluation imports, supplies bundled official FastChat preprocessing and a
compatible Arrow reader, and forces right padding for the assistant label mask.
The generated DeepSpeed config renames `bfloat16` to `bf16` and removes the
original DS WarmupDecayLR so HF actually applies the requested cosine schedule.
All other original DeepSpeed settings are retained. This is original IF-SFT
training with the stated adaptations, not a byte-identical historical environment
or reproduction guarantee for its published checkpoint.

`scripts/setup_tfa_colab.py --original-if` installs the scoped compatible stack
and DeepSpeed 0.18.8; it rejects pip plans that modify installed Torch, CUDA,
Triton, torchvision or torchaudio. It checks imports, the actual slow tokenizer,
fingerprint supervision and builds CPUAdam before training. It does not require
bitsandbytes or TRL. Compatible version pins are intentional: latest major
Transformers releases remove APIs used by the old trainer.

**Resource limit:** CPU Adam's FP32 master weights and two moments alone require
about 75 GiB for Llama2-7B. Gradients and buffers add more. The user's 83 GiB
Colab may run out of RAM. This strict CPU-offload configuration does not solve
that limit, and has not been validated on a full Colab GPU run.

Start with a fresh output directory; existing output is refused. Training
produces `checkpoint/`, `recipe.json`, and `deepspeed.json`. Evaluation/MEraser
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
!python -u scripts/setup_tfa_colab.py --original-if && python -u scripts/train_original_if_tfa.py --output-dir outputs/original_if_tfa_v1
```

Local verification: 119 CPU tests passed, including a real tiny Llama train/
export through the original training body and actual HF DeepSpeed configuration
resolution for batch 60/LR/weight decay. Local Torch was 2.5.1+cpu, Transformers
4.46.0 and Accelerate 1.0.1; these do not validate Colab's GPU kernels or pinned
production stack. The cached fast tokenizer was used only for the CPU training
test on two supervised normal rows; production uses the slow tokenizer and
checks all fingerprint rows during setup and again before loading the 7B model.
