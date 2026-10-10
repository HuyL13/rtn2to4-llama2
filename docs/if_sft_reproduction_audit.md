# IF-SFT Llama2-7B reproduction audit (2026-10-10)

## What the sources establish

The [official configuration](https://github.com/cnut1648/Model-Fingerprint/blob/main/configs/sft_chat.yaml)
specifies `NousResearch/Llama-2-7b-hf`, 3 epochs, LR `2e-5`, total batch 64.
The [launch pipeline](https://github.com/cnut1648/Model-Fingerprint/blob/main/pipeline_SFT_chat.py)
uses microbatch 4, BF16 mixed precision, checkpointing, cosine CLI argument,
weight decay .01 and seed 42. Its [DeepSpeed JSON](https://github.com/cnut1648/Model-Fingerprint/blob/main/deepspeed_config/zero3-offload.json)
supplies Adam and **WarmupDecayLR**, which takes precedence over the CLI cosine.
The training script leaves loading dtype implicit (FP32); the previous wrapper
instead loaded BF16 and used paged Adam8. These are material differences.

[The IF paper, section 4.3](https://arxiv.org/html/2401.12255v2#S4.SS3)
describes dialogue training and assistant-only loss. The precise 3/2e-5/64
recipe comes from the repository, not an independently documented historical
Llama2 checkpoint training log. We cannot claim bitwise reproduction.

All eight direct GitHub forks were audited using the public forks/compare APIs.
None changes the SFT configuration. Ardeur-HK changes a GPT-J adapter batch;
jacobzhuu adds a Mistral adapter experiment. Neither provides an independently
successful Llama2 full-SFT recipe. Official issues
[#3](https://github.com/cnut1648/Model-Fingerprint/issues/3) and
[#5](https://github.com/cnut1648/Model-Fingerprint/issues/5) report reproduction
problems and have no resolution in their comments.

* [MergeGuard section 3.3](https://arxiv.org/html/2404.05188v1#S3.SS3)
  reports successful SFT insertion into **Llama2-7B-CHAT**, VSR 1.000. Its
  [code](https://github.com/CryptoAILab/MergeGuard) does not disclose the insertion
  recipe. CHAT is a different starting model and is not silently substituted.
* [FPEdit appendix A.5](https://arxiv.org/html/2508.02092v2#A5)
  uses public Llama2 and Mistral fingerprinted checkpoints. Its three-epoch
  reproduction concerns Llama3 and GPT-J, not independent Llama2 retraining.
* [The Challenge of Identifying the Origin of Black-Box LLMs](https://arxiv.org/html/2503.04332v1)
  reports IF Llama2 response rate 1.00 but does not disclose its IF insertion
  recipe or establish a single-A100 full-SFT reproduction.

## Observations and hypotheses

The user's v2 run has native FSR 0%, PPL 7.4318; the public checkpoint has FSR
100%, PPL 7.5013. Finishing six optimizer steps is not evidence of successful
fingerprint insertion. With 128 rows and batch 64, three epochs really are just
six updates.

Two separate effects require measurement:

1. Direct BF16 parameter updates can round away at small LR. A CPU AdamW toy
   experiment starting at .1 leaves the BF16 parameter unchanged after six
   steps of LR 2e-5; an FP32 master changes. This is a mechanism demonstration,
   **not causal proof for the user's 7B run**. The new precision monitor observes
   the actual DeepSpeed CPUAdam master updates.
2. Upstream FastChat preprocessing maps `human` to `USER`, while upstream
   inference appends literal `human` as the role. The audit scores the same
   eight pairs with both prompts, same target/prefix and decoding settings.
   Neither prompt result silently replaces native FSR. An eight-row diagnostic
   is recall on training fingerprints, not held-out generalization.

The existing ignored normal row (empty human input) and the upstream negative
example containing the builtin `hash` string are retained. They should be
reported, not silently rewritten when claiming upstream reuse.

## Closest supported resource adaptation

`--profile colab_nvme` reuses upstream `run_chat.py`, data, loss, all-parameter
training, microbatch 4 / accumulation 16, LR, epochs, Adam and its scheduler.
It restores the FP32 loading request and delegates master/moment handling to
DeepSpeed. ZeRO-3 initialization itself chooses BF16 from its configuration;
the loading flag does not promise unrounded FP32 source values in GPU weights.
The important verified property is FP32 masters for subsequent updates.
Optimizer **storage**, subgroup/bucket sizes and checkpoint policy change.
The legacy JSON `bfloat16` key becomes canonical `bf16` so HF and DeepSpeed
recognize the same mixed precision mode.
DeepSpeed 0.19.7 rejects a zero `warmup_num_steps`, whereas 0.12.6 silently
clamped it to two. Both the tiny preflight and real DeepSpeed training now pass
`warmup_steps=2` to HF so the JSON's `auto` resolves consistently. This is a
documented compatibility adjustment: modern DeepSpeed also initializes LR to
zero, while 0.12.6 left the initial optimizer LR unchanged. The complete
historical LR trajectory is therefore not claimed to be bitwise identical.
See [DeepSpeed ZeRO-Infinity](https://deepspeed.readthedocs.io/en/stable/zero3.html)
and [memory accounting](https://deepspeed.readthedocs.io/en/stable/memory.html).
Forward/backward still use mixed BF16; CPUAdam updates FP32 masters and states.
This is closer to upstream than changing optimizer or switching to LoRA.

FP32 master plus two Adam moments require about 75 GiB; accumulated gradients
can add another 25 GiB. CPU offload alone has insufficient safe headroom in an
83-GiB Colab runtime. NVMe here means the local VM disk; Drive FUSE is rejected.
The wrapper requires 160 GiB free before loading 7B. It checks the installed
CPUAdam/AIO native extensions and runs a tiny real Trainer train/save/resume
with NVMe before allocating 7B. Train and resume each get a fresh standalone
worker process so Accelerate's global DeepSpeed plugin state cannot collide.
The resumed final model is checked against the uninterrupted final model.
Missing DeepSpeed/libaio/compiler is reported;
the runner never installs libraries or replaces Torch/CUDA.

To fit local disk, the **7B run exports only the final HF model**, without the
75-GiB optimizer checkpoint copies. An interrupted six-step training run must
start training again. A completed run resumes evaluation from its marker.
`nvme_swap/` is working storage, not a portable resumable training checkpoint.
Keep it while training; do not upload it to GitHub. Keep the final checkpoint
on persistent storage if the Colab VM will be deleted.

The local development environment has no CUDA, DeepSpeed, SentencePiece or
bitsandbytes. CPU regression tests can validate configuration, prompt pairing,
and the precision mechanism, but cannot establish 7B convergence or GPU fit.
`nvme_preflight.json`, `checkpoint/precision_monitor.json`, `checkpoint/train_results.json`,
`prompt_audit/summary.json` and native baseline/PPL reports are the runtime evidence.

## Colab

Existing environment: DeepSpeed with working CPUAdam and async I/O, the
IF-SFT/MEraser requirements, a BF16 GPU, and HF_TOKEN in Secrets. No installer
is embedded in the training script. First inspect disk; missing resources stop
before loading 7B and existing results are not deleted.

```python
%cd /content/rtn2to4-llama2
!git pull --ff-only
!df -h /content
import os
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
!python scripts/if_sft_fidelity.py --model-path outputs/llama2_if_sft_retrained_v2/checkpoint --output-dir outputs/llama2_if_sft_retrained_v2/prompt_audit
!bash run_retrain_if_sft.sh --profile colab_nvme --output-dir outputs/llama2_if_sft_fp32_v5
```

The first audit line requires the existing v2 checkpoint; omit it in a fresh
runtime without that checkpoint. The new run audits prompts automatically,
evaluates native FSR and C4 PPL, and starts the existing MEraser pipeline only
after native FSR passes 95%. English Random, Perinucleus, CTCC and old IF-SFT
training are not invoked.
