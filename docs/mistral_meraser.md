# Online Mistral IF-SFT evaluation

Use `cnut1648/Mistral-7B-fingerprinted-SFT` with the original
`Model-Fingerprint/dataset/llama_fingerprint_chat` Arrow dataset and
`method: if_sft` in `source_reference.json`.

MEraser evaluation loads the BF16 source and optional LoRA adapter onto the
current CUDA GPU explicitly. CPU offload is disabled; insufficient VRAM raises
an error rather than silently slowing generation. Close other GPU workloads
before evaluating. The runner does not install or replace Torch.

Native FSR generation evaluates the first eight validation rows, which are
the eight rows used by the upstream FSR calculator. It retains the original
prompt, greedy decoding, 30-token budget, and native substring FSR calculation.
Normal/regularization rows are omitted; this is not a normal-prompt robustness
evaluation. KV caching is enabled for MEraser generation, matching upstream
inference. Other callers of `evaluate_fingerprint` retain their cache default.
C4 PPL remains 8 blocks of 2048 tokens and disables cache while scoring.

After interrupting a slow baseline evaluation, pull the update and rerun
`--worker eval_base` with the same method/output directories; partial prediction
files are regenerated. Then run `erase`, `eval_erase`, `merge`, `recover`, and
`eval_recover` as separate workers. The explicit workers allow measuring recovery
even when erasure has not reached zero FSR. Neither IF-SFT training nor a different
model checkpoint is required.

Local validation: 127 CPU tests passed, including ten temporary regression cases
covering all three evaluation stages with Llama/Mistral sources, unchanged native
FSR after omitting regularization rows, cache configuration propagation, and
identical cached/uncached greedy output on tiny real Llama and Mistral models.
Full 7B GPU throughput was not tested locally.
