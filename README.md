# Llama-2-7B: direct RTN4 versus coarse-locked Nested 2→4

All newly trained fingerprint checkpoints use `meta-llama/Llama-2-7b-hf`.
IF-SFT reuses `cnut1648/LLaMA2-7B-fingerprinted-SFT` from the original repo.
Natural-language query construction for ImF uses `meta-llama/Llama-2-7b-chat-hf`;
the ADG carrier and trained victim both use the base Llama-2-7B.

## Run on the server

Scalable training now reuses the Phase A GPU paged AdamW 8-bit recipe, with
averaging and benign mixing disabled. Default fingerprint count is 64, and
gradient accumulation follows the upstream full-batch rule. CTCC uses BF16
and effective batch 16; ImF stays on its local DeepSpeed recipe. See
[integration and Colab commands](docs/phasea-training-integration.md).
These settings differ from the paper recipe. Use a new output directory for
new runs; completed IF-SFT results remain available.

Copy this complete folder (including `vendor/`, `upstream/`, and `Model-Fingerprint/`) to the server, then run:

```bash
cd rtn2to4
CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh
```

The script **does not install packages, clone repositories, create or activate environments**.
It uses the existing Python/environment. Set `PYTHON_BIN=/path/to/python` if needed.
The existing Hugging Face login or `HF_TOKEN` must have access to Meta's base/chat Llama-2 models.
The main training path is for one CUDA GPU supporting BF16, with A100 40GB or greater and sufficient CPU RAM for optimizer offload/model averaging. Four new dense source checkpoints need roughly 56GB of weight storage, plus training optimizer checkpoints, datasets/cache, and one temporary ~14GB evaluation export. Reserve substantially more disk for training state.

Server dependencies include CUDA PyTorch, Transformers **4.46.0–4.46.1**, datasets, accelerate, DeepSpeed, PEFT, wandb, lm-eval, sentencepiece, and LLaMA-Factory v0.9.1 dependencies. LLaMA-Factory is included and selected through its `src/train.py`, not an installed framework of a different version. Its compatibility ranges are in `upstream/LLaMA-Factory/requirements.txt` (accelerate <=1.0.1, PEFT <=0.12.0, TRL <=0.9.6, numpy <2). Configure these separately; no setup commands are in the run script.

## Run on Google Colab

Before loading checkpoint weights, the runner now checks actual imports in isolated
processes for the selected methods: FastChat/IF-SFT, lm-eval ARC, scalable training,
ImF, LLaMA-Factory/TRL and DeepSpeed. It also runs a tiny Llama forward/backward on
CPU and, if available, the runtime CUDA GPU. Run it independently with
`python scripts/check_environment.py --require-cuda`. IF-SFT uses the bundled official
FastChat v0.2.36 conversation code and VicunaAdapter template selection, so it does not
require installing `fschat` or importing model adapters. Dependency ranges are in
`requirements-experiment.txt`.
TensorFlow and Flax backends are disabled automatically for this PyTorch pipeline.
IF-SFT dataset loading handles the newer `List` metadata in the existing Arrow
files when using datasets 3.1.0: the compatibility loader reads the same saved
rows and rebuilds feature metadata. Conversation order, splits and labels are
preserved. Preflight now loads all three splits before downloading model weights.
Full training, DeepSpeed optimizer compilation, and Python 3.13 compatibility still
need verification on the actual server/Colab runtime; CPU tests cannot establish these.

Scalable training uses `report_to="none"` explicitly. Transformers interprets
`None` as enabling installed integrations, which imported TensorBoard and then
the incompatible TensorFlow/JAX stack on Colab. The environment checker now
also runs two tiny training steps using the actual scalable Trainer, fingerprint
and benign collators, gradient accumulation, evaluation, model averaging and
HF export/reload. It asserts that no TensorFlow backend is imported.
To additionally check CUDA DeepSpeed stage 2 and CPUAdam JIT compilation before
loading the 7B model, run `python scripts/training_smoke.py --launch-deepspeed`.
This entry point sets backend environment variables before starting the DeepSpeed
launcher, which itself imports Transformers before the training script starts.
This creates a temporary tiny model, downloads no weights, installs no packages,
and does not change experiment checkpoints or results.

Scalable epoch averaging now keeps its immutable original parameters on disk,
loading one parameter with memory mapping at a time. The original full-parameter
multiply/add operations and coefficient are preserved; CPU tests checked bitwise
agreement for FP32, FP16 and BF16 over multiple epochs. This removes the full
~13GB resident CPU model copy. Allow another ~13GB disk per scalable method under
`training/saved_models/<hash>/averaging_reference/`.
DataLoader workers are set to zero to avoid forking a process holding a large
CPU-offloaded optimizer. DeepSpeed communication buckets are limited to 5 million
elements. AdamW, full fine-tuning and the objective are unchanged. RAM usage is
printed around averaging and Trainer initialization. CPU optimizer state still
requires substantial RAM; a tiny smoke test cannot establish 7B peak memory.

The exact Colab cells use one command per line, with `!` for shell commands and `%cd` for
directory changes. Copy them from [`docs/colab_commands.md`](docs/colab_commands.md). A GPU
runtime, Hugging Face access to both Llama-2 checkpoints, and enough VRAM/RAM are required.

Inspect without loading models or changing files:

```bash
bash run_new_experiment_llama2.sh --dry-run
```

Run a subset:

```bash
bash run_new_experiment_llama2.sh --methods if_sft
bash run_new_experiment_llama2.sh --methods english_random perinucleus
bash run_new_experiment_llama2.sh --methods imf ctcc
```

Rerunning the same command reuses complete source references and complete variant results.
Interrupted scalable exports are preserved under `final_model.incomplete-*` and retrained.
Interrupted/empty fingerprint JSON files are preserved as `*.incomplete-*` and
regenerated automatically. English key generation now uses `max_new_tokens`
instead of an absolute `max_length`, which failed for one-token responses with
Llama-2's prompt/BOS tokens. Training still truncates keys to 16 tokens and uses
the existing one-token target construction. Completed datasets and evaluation
results are reused by rerunning the same command and output directory.
Use a new `--output-dir` when changing training/evaluation settings; input hashes prevent mixing datasets/settings.
`--skip-training` requires saved source references for all selected new methods.

For a server smoke test, use a separate directory; it still trains the selected source unless IF-SFT is selected:

```bash
bash run_new_experiment_llama2.sh --methods if_sft \
  --ppl-max-tokens 2048 --lm-eval-limit 2 \
  --output-dir outputs/llama2_smoke
```

Default evaluation: source full precision, RTN2, direct RTN4, Nested 2→4.
Add `nested_2to3` explicitly for the ablation. `fp_base` is required for the source gate.
Default dtype is BF16, matching the original experiment shell; `--dtype fp16` selects FP16 evaluation consistently across PPL, fingerprint evaluation and ARC. The source metric is labelled `fp_base`, not FP16 when evaluated in BF16.

## Training and native metrics

| Method | Training/data reused | Primary fingerprint metric |
|---|---|---|
| IF-SFT | Original public checkpoint and local Model-Fingerprint validation data | Upstream flexible FSR; exact FSR and target NLL retained |
| English-Random | SewoongLab full fine-tuning, English keys and random one-token responses | Upstream one-token greedy recall, cross-checked against argmax |
| Perinucleus | SewoongLab generation on Llama-2, nucleus threshold 0.8 and k=3; full fine-tuning | Same top-1 recall as its paper |
| ImF | Latest local native ADG code; create new Llama-2 targets and train a new source | Decoded ownership payload equals registered message; clean-base false verification measured separately |
| CTCC | Official trigger/suppression/normal datasets, LLaMA-Factory LoRA training, dense PEFT merge | Exact trigger FSR and negative false activation FSR |

English-Random and Perinucleus default to 1024 fingerprints, key length 16, response length 1, 30 epochs, LR 5e-5, weight averaging 0.75, benign fraction 0.25, seed 42. Batch size 4 preserves at least one benign sample per batch in the upstream implementation. Both evaluate the actual saved training pairs, never freshly resampled random responses.

CTCC uses its default Llama-2 recipe: 12 epochs, LR 1e-4, LoRA rank 8/alpha 16, batch 2 with accumulation 8. The released test set is partitioned by the `IAMALIVE` target into 95 triggers and 205 negatives. Each group uses its own denominator. The Llama-2 prompt template and greedy generation follow the upstream evaluation code; scoring follows the paper's exact-response definition, rather than the substring test in its pruning script. Leading/trailing whitespace is stripped; case and punctuation are preserved. This runner avoids an extra automatic BOS because the upstream template already supplies it.

ImF uses the local reconstructed ADG experiment. **It is not an exact reproduction of the authors' unavailable original decoder/key**, or the paper's full iterative query refinement/manual selection. The inherited query builder requests exact reference reproduction. Targets, message/key, carrier revision, and construction metadata are kept together in the manifest; successful unit tests do not establish scientific equivalence with the paper. Clean base must have zero decoded payload successes; the trained source must pass the native source gate. If either fails, the runner stops before quantization, without post-PTQ query tuning. This limitation is inherited from the selected local implementation.

Source native FSR must be >=95% before quantized variants run. With ten ImF fingerprints this requires 10/10 successes. Training convergence is checked on the server; no GPU results are bundled.

## Quantization and utility

The existing per-output-channel RTN implementation is reused unchanged. All variants start independently from the same source, and use the same seven selected projection modules. Embeddings, norms, and lm_head follow the original exclusion policy. Nested refinement reads the original FP source and restricts each weight to four midpoints inside its original RTN2 cell; it does not attempt to infer missing residual information from RTN2 weights alone.

Runtime audits check every selected weight **after dtype conversion**, against original RTN2 min/max and assignments, cell boundaries, and allowed fine levels. Any violation aborts. Successful statistics report zero violations; no quantizer hyperparameters are tuned per fingerprint.

PPL reuses the original `evaluate_ppl_current_model` and `eval_ppl.py`; defaults are C4, sequence length 2048, max 16384 tokens, as in the original run script. lm-eval uses the unmodified local RBVT runner plus a small dtype adapter. It runs **only `arc_challenge` and `arc_easy`**, default zero-shot with no sample limit. Temporary dense quantized-on-grid exports ensure the actual modified weights are evaluated. They are not packed INT4 model files. Each evaluator runs after the quantization worker exits, avoiding simultaneous copies of the victim model on the GPU.

Results are under `outputs/llama2_new_experiment/<method>/evaluation/`: `results.json`, `results.csv`, and per-variant native predictions, quantization statistics in `result.json`, and lm-eval raw results. `--lm-eval-limit` results must be described as smoke/subsample results.

## Code provenance

- Original rtn2to4: `613fe8e`.
- Local RBVT-squeeze: `a362565080e9151c6d779cd93e567c7edbb0c96d`; `lm_eval_runner.py` and `runtime_utils.py` copied byte-for-byte.
- Local ImF: `imf_ptq` at `4a5ed38`; native codec, generator, metrics, evaluator and CAU trainer copied into `vendor/imf_native`. Adaptations: Llama-2 backbone pin, Llama-2 chat helper, evaluation dtype, preserve carrier tokenizer vocabulary, and Trainer's DeepSpeed-aware model export. The original local folder is unchanged.
- SewoongLab official source: `fdceaba14bd3e89340916a6a40e27c945d48460e`, preserved in `upstream/scalable-fingerprinting-of-llms`; runnable copy in `vendor/scalable`. Compatibility changes: remove redundant deprecated TrainingArguments alias, enable gradient checkpointing, load the supplied model in BF16, remove unnecessary model-unwrapping of the tokenizer; copy the existing random signature file to the path expected by the dataloader. Generation, objective and evaluator are reused.
- CTCC official source: `8db93218260bed31b8f18acc9c6ac3e1955d3a42`, preserved under `upstream/CTCC`.
- LLaMA-Factory v0.9.1: `57354fc9904fe36daa6ddd9010d4936d84188ddb`, pinned for the local Transformers 4.46 API.

Paper metric references: [Scalable Fingerprinting](https://arxiv.org/html/2502.07760v2), [CTCC](https://aclanthology.org/2025.emnlp-main.356/), [ImF](https://aclanthology.org/2026.acl-long.1183/).

CPU checks can be run in the prepared development environment:

```bash
PYTHONPATH=.:vendor/imf_native:vendor/imf_native/src python -m pytest tests vendor/imf_native/tests -q
bash -n run_new_experiment_llama2.sh
```
