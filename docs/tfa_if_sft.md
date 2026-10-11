# TFA/SVA upstream IF-SFT on Llama2-7B

Source: https://github.com/CAU-ISS-Lab/LLM-Fingerprint-and-Attacks/tree/b5ac14cf7086e700a02892a89d375c2d042684e1/TFA_SVA

Vendored files under `upstream/TFA_SVA` are unchanged; `provenance.json`
records their SHA256 hashes and revision. The launcher verifies them before
running. Its `utils` binding prevents importing an unrelated local module.

The upstream trainer uses full-parameter SFT, Alpaca instruction prompts,
target-only labels, EOS targets, dynamic right padding, and its own JSON data.
`train_IF_60.json` contains 60 rows, including 10 fingerprint rows. Its native
default maximum length is 512. This is a different dataset/prompt recipe from
the earlier Model-Fingerprint dialogue/128-row experiment; its results must
not be evaluated using the old eight keys or passed to the existing MEraser
IF-SFT evaluator unchanged.

Upstream `Fingerprint_dataset/run.sh` targets Qwen2.5-7B on four GPUs, batch
15 per GPU, accumulation 1, 20 epochs, LR 2e-5, weight decay 0, warmup ratio
0.03 and cosine scheduling. This launcher changes the backbone to
`NousResearch/Llama-2-7b-hf` (the same Llama2 base mirror used previously), and
uses batch 1 with accumulation 60 on one GPU: the effective batch remains 60.
Thus 60 rows x 20 epochs / batch 60 yields 20 optimizer steps. It preserves
the upstream trainer, dataset, collator, loss, default length and scheduler.

Tokenizer detail: the upstream special-token branch checks the case-sensitive
substring `"llama"`. The mirror ID contains `Llama`, so that branch is skipped
unchanged and the mirror's existing BOS/UNK tokens are retained. We do not
silently lowercase the condition or redefine those tokens. Using a lowercase
model path would execute a different upstream branch.

`--profile upstream` keeps the original optimizer/offload/precision configuration
and save function. Its runtime copy adds only `gradient_accumulation_steps:
"auto"` to synchronize the one-GPU accumulation=60 setting with Hugging Face.
The original JSON omits that key, so DeepSpeed otherwise defaults to 1 and
rejects `train_batch_size=60` before loading the model. The vendored JSON is
unchanged. This profile does not reproduce the four-GPU topology and is not claimed to
fit Colab RAM. Upstream requirements are an environment dump containing local
Conda paths and explicit Torch wheels; never install that file wholesale.

`--profile colab_nvme` makes these explicit deviations:

* BF16 compute, FP32 Adam masters, CPUAdam instead of Torch AdamW; weight decay
  remains zero with the same default betas and epsilon. Numerical identity is
  not claimed.
* NVMe optimizer offload, CPU parameter offload, bounded communication buckets
  and disabled overlap. Uses official DeepSpeed 0.18.8 to avoid 0.19.7's mlock
  allocator under Colab's 8 MiB memlock cap.
* Final export uses `Trainer.save_model()` and the upstream ZeRO-3 gather flag,
  rather than copying partitioned state_dict tensors. The tokenizer is exported
  by Trainer too. Intermediate optimizer checkpoints are disabled to avoid
  duplicating the swap state; interrupted training cannot resume.
* Requires a fresh output folder and at least 160 GiB free on local disk. Never
  reuse a live or partial NVMe run directory. This prevents stale swap reuse;
  it does not prove the earlier short-file failure was caused by stale files.

Training source is compiled unchanged into an isolated module. Only the final
save function is replaced for Colab. There is no LoRA, 8-bit optimizer,
Adafactor, label rewrite, dataset regeneration, tiny train/resume comparison,
or automatic old-IF-SFT/MEraser evaluation in this launcher.

## Colab

Select A100 GPU and configure HF_TOKEN in Colab Secrets. Use a fresh runtime.

```python
%cd /content
import os, subprocess, sys
from pathlib import Path
from google.colab import userdata
repo=Path('/content/rtn2to4-llama2')
if repo.exists():
    subprocess.run(['git','-C',str(repo),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/HuyL13/rtn2to4-llama2.git',str(repo)],check=True)
os.environ['HF_TOKEN']=userdata.get('HF_TOKEN')
os.environ['MAX_JOBS']='2'
%cd /content/rtn2to4-llama2
subprocess.run([sys.executable,'scripts/setup_tfa_colab.py'],check=True)
subprocess.run([sys.executable,'scripts/train_tfa_if_sft.py','--profile','colab_nvme','--output-dir','outputs/tfa_if_llama2_20e'],check=True)
```

Setup is explicit and separate from training. It resolves wheels first, rejects
any planned Torch/CUDA replacement, and installs DeepSpeed with no dependencies
or isolated build. It checks native compilation, large pinned buffers and AIO
read/write before loading 7B. Training still needs actual Colab validation;
CPU tests do not establish GPU/NVMe stability or fingerprint quality.

Logs show `N/20`, loss and epoch. Model/tokenizer are written to
`outputs/tfa_if_llama2_20e/checkpoint`. Run with `--dry-run` to inspect the recipe
without allocating a model. Do not run setup while another training job is live.
