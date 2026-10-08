# Phase A training integration

Source: https://github.com/huynguyenquang-collab/phaseA_fingerprint
Revision: a61c314. Reuse its documented scalable recipe: paged AdamW 8-bit,
no DeepSpeed, no epoch averaging, no benign mixing, and upstream full-batch
gradient accumulation. Keep Llama-2-7B, native evaluators, PPL, ARC-C/E, and
RTN2/RTN4/Nested 2-to-4 in this project. ImF stays on its local recipe.
Original training scripts and the optimizer patch are copied unchanged under
`upstream/phaseA_training/` for provenance; the pipeline does not execute them.
The existing pinned scalable modules are reused with equivalent training flags.
We use microbatch 4 for Colab headroom rather than Phase A's 8.

Implementation checks: verify recipe/CLI and response-file selection on CPU;
run real paged optimizer training on a tiny CUDA Llama before loading 7B;
run existing regression tests. CPU-only checks cannot validate CUDA training.

CTCC reuses BF16 and microbatch 4 with accumulation 4 (effective batch 16).
Keep pinned local LLaMA-Factory, local datasets/template and merge/evaluator.
Do not enable Liger without an installed compatible dependency. Do not use
Phase A setup scripts, torchaudio stubs, automatic pushes or checkpoint deletion.

Perinucleus trains on the generator's derived response file, not the key file.
Generate twice as many keys and retain the first requested number of valid,
non-special single-token responses. Record rejected counts. Refuse to train if
not enough valid fingerprints exist; never silently reduce the denominator.

Scalable runs use a new output directory because optimizer, averaging, mixing
and effective batch change. These changes follow Phase A, not an exact paper
reproduction. Default fingerprint count becomes 64; use --num-fingerprints 1024
if needed. All configurations and the source provenance are recorded.

Use Colab with HF_TOKEN set and compatible bitsandbytes already installed:

```text
%cd /content/rtn2to4-llama2
!git pull --ff-only
!python scripts/check_environment.py --methods english_random perinucleus --require-cuda
!CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh --methods english_random perinucleus --output-dir outputs/llama2_phasea_v3 --num-fingerprints 64
```

CTCC and ImF can be run separately in the same new output root:

```text
!CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh --methods ctcc imf --output-dir outputs/llama2_phasea_v3
```

The checker trains a tiny CUDA Llama with the actual paged AdamW optimizer and
checks 8-bit states before loading 7B. Source FSR must still reach 95%. No
automatic 64-to-256 fallback is added: changing the count requires an explicit
new run/output directory. Training exports only final scalable weights, so
interrupted scalable training restarts; it cannot resume an epoch checkpoint.
