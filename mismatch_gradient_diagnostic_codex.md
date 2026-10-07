# Mismatch-Gradient Diagnostic for Fingerprint Erasure
## Implementation specification for Codex

## 0. Goal

Before implementing any new quantization algorithm, run a **cheap full-precision diagnostic** to answer one question:

> Does a generic clean-vs-mismatched causal-LM gradient provide a direction that weakens a model fingerprint before benign language-model utility collapses?

This experiment is **NOT quantization**.

Do not implement PTQ, LNQ, GuidedQuant, MaxDrift, pseudo-keys, or any fingerprint-aware optimization in this stage.

The output of this stage is only a curve showing how:

- benign perplexity,
- fingerprint Flexible FSR,
- fingerprint Exact FSR,
- fingerprint target NLL,

change when the fingerprinted FP model is perturbed along one generic mismatch-derived gradient direction.

If this direction does not affect the fingerprint before utility degrades badly, stop this research branch and do not build a quantizer around it.

---

# 1. High-level experiment

Starting from one fingerprinted FP model \(W\), first test **IF-SFT only**.

Use benign text sequences to construct:

1. a clean causal-LM continuation task;
2. a mismatched-context causal-LM continuation task.

Compute:

\[
L_{\text{clean}}
\]

and

\[
L_{\text{mis}}.
\]

Define the contrastive erasure loss:

\[
\boxed{
L_E = L_{\text{mis}} - L_{\text{clean}}
}
\]

and its gradient:

\[
\boxed{
g_E = \nabla_W L_E
    = \nabla_W L_{\text{mis}}
    - \nabla_W L_{\text{clean}}.
}
\]

Do not train the model.

Compute this gradient once, then create a sequence of temporary perturbed models:

\[
\boxed{
W_\eta = W - \eta g_E.
}
\]

Evaluate each perturbation level.

The purpose is to test whether this generic direction is a meaningful **association-destruction direction**.

---

# 2. Hard constraints

The diagnostic MUST satisfy all of the following.

## 2.1 No quantization

Do not call:

- RTN;
- GPTQ;
- AWQ;
- LNQ;
- GuidedQuant quantization;
- MaxDrift;
- any integer-code optimizer.

The model remains FP16/BF16 throughout this diagnostic.

## 2.2 No fingerprint information in gradient construction

The gradient dataset must contain only benign text.

Do not use:

- fingerprint keys;
- fingerprint triggers;
- fingerprint responses;
- fingerprint prompts;
- fingerprint training samples.

Fingerprint data is used only during final evaluation after the gradient has been computed.

## 2.3 No optimization loop over model weights

There is only one gradient computation.

Do not run SGD/Adam.

Do not repeatedly optimize \(L_E\).

Do not fine-tune.

The only model modifications are direct diagnostic perturbations:

\[
W \leftarrow W - \eta g_E.
\]

## 2.4 Perturb only quantizable transformer linear weights

Apply the diagnostic gradient and perturbation only to the weights that would later be PTQ-quantized.

For a Llama-style model, initially include:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Exclude:

```text
token embeddings
lm_head
LayerNorm / RMSNorm
biases
```

This keeps the diagnostic aligned with the later PTQ search space.

---

# 3. Benign calibration data

Use a small fixed benign subset for this diagnostic.

Recommended first run:

```text
dataset: C4
number of sequences: 32
sequence length: 512 tokens
prefix length: 256 tokens
continuation length: 256 tokens
seed: 42
```

Use exactly the same 32 source sequences for clean and mismatched construction.

Do not use fingerprint data.

Cache the tokenized sequences so repeated runs use exactly the same samples.

---

# 4. Construct clean and mismatched causal-LM sequences

For each token sequence:

\[
s_i = [a_i ; b_i]
\]

where:

- \(a_i\): first 256 tokens;
- \(b_i\): final 256 tokens.

Create a deterministic derangement.

For the first implementation, use a simple cyclic shift:

\[
\boxed{
\pi(i)=(i+1)\bmod N.
}
\]

This guarantees:

\[
\pi(i)\ne i
\]

for all samples.

## 4.1 Clean sequence

For sample \(i\):

\[
s_i^{\text{clean}}=[a_i;b_i].
\]

## 4.2 Mismatched sequence

For the same continuation \(b_i\):

\[
\boxed{
s_i^{\text{mis}}=[a_{\pi(i)};b_i].
}
\]

Therefore clean and mismatched versions have:

- exactly the same continuation target \(b_i\);
- the same total sequence length;
- different preceding context only.

This is important.

The experiment must not accidentally change the continuation target between clean and mismatched examples.

---

# 5. Loss masking

The loss must be computed **only on continuation tokens** \(b_i\).

For both clean and mismatched sequences:

```text
labels for prefix tokens       = -100
labels for continuation tokens = actual token ids
```

Use standard causal-LM next-token loss.

Thus:

\[
L_{\text{clean}}
=
-\log p_W(b_i\mid a_i)
\]

and

\[
L_{\text{mis}}
=
-\log p_W(b_i\mid a_{\pi(i)}).
\]

Do not include the prefix tokens in the loss.

---

# 6. Compute the mismatch gradient efficiently

Do NOT keep clean and mismatched computation graphs in memory simultaneously.

Use two sequential gradient-accumulation passes.

Initialize all selected gradients to zero.

First accumulate:

\[
-g_{\text{clean}}.
\]

Then accumulate:

\[
+g_{\text{mis}}.
\]

At the end:

\[
\boxed{
g_E = g_{\text{mis}}-g_{\text{clean}}.
}
\]

A memory-efficient implementation is:

```python
zero_grad()

# Pass A: clean term with negative sign
for batch in clean_loader:
    loss_clean = continuation_only_loss(model, batch)
    (-loss_clean / num_batches).backward()

# Pass B: mismatched term with positive sign
for batch in mismatch_loader:
    loss_mis = continuation_only_loss(model, batch)
    (loss_mis / num_batches).backward()

# parameter.grad now contains g_E
```

If batch sizes differ or the last batch is incomplete, normalize by the exact number of continuation tokens rather than `num_batches`.

For the default `N=32`, choose a batch size that divides 32.

Try in this order:

```text
batch_size = 4
batch_size = 2
batch_size = 1
```

Use the largest one that fits memory.

---

# 7. Model mode and memory settings

Use:

```python
model.config.use_cache = False
```

Only selected transformer linear weights should have:

```python
requires_grad = True
```

Everything else:

```python
requires_grad = False
```

No optimizer object is needed.

No optimizer states should be allocated.

After gradient computation:

- free temporary activations;
- call `torch.cuda.empty_cache()` once if useful;
- retain only model weights and the computed gradients.

Do not move gradients CPU <-> GPU inside the perturbation/evaluation loop.

---

# 8. Gradient statistics

Before perturbing the model, compute over all selected weights:

\[
\|W\|_2
=
\sqrt{
\sum_p\|W_p\|_F^2
}
\]

and

\[
\|g_E\|_2
=
\sqrt{
\sum_p\|g_{E,p}\|_F^2
}.
\]

Log:

```text
selected_parameter_count
weight_norm
gradient_norm
clean_loss
mismatch_loss
mismatch_minus_clean_loss
```

Also log per-layer gradient norms for diagnosis, but do not use them to select layers.

Do not rank layers and do not perturb only top-k layers.

---

# 9. Perturbation scale

Do not choose an arbitrary learning rate \(\eta\).

Specify the perturbation by target relative parameter drift:

\[
\boxed{
r
=
\frac{\|W_\eta-W\|_2}{\|W\|_2}.
}
\]

Because:

\[
W_\eta-W=-\eta g_E,
\]

choose:

\[
\boxed{
\eta(r)
=
r\frac{\|W\|_2}{\|g_E\|_2}.
}
\]

Default diagnostic grid:

```text
r ∈ {
    3e-4,
    1e-3,
    3e-3,
    1e-2
}
```

Also evaluate the unmodified model:

```text
r = 0
```

Important:

FP16/BF16 arithmetic may cause very small elementwise perturbations to round away.

Therefore after applying each perturbation level, compute and log the **actual measured relative drift** if possible.

Do not assume the requested drift exactly equals the realized drift.

---

# 10. Apply perturbations cumulatively for speed

Do not reload or clone the 7B model four times.

Sort target drift levels ascending:

```text
0
3e-4
1e-3
3e-3
1e-2
```

After computing \(g_E\), keep the same model in memory.

For target \(r_k\), compute:

\[
\eta_k=r_k\frac{\|W\|}{\|g_E\|}.
\]

Move from the previous level to the next using:

\[
\boxed{
W_{\eta_k}
=
W_{\eta_{k-1}}
-
(\eta_k-\eta_{k-1})g_E.
}
\]

In code:

```python
delta_eta = eta_k - eta_prev

with torch.no_grad():
    for p in selected_params:
        p.add_(p.grad, alpha=-delta_eta)
```

Then evaluate immediately.

This avoids:

- multiple checkpoints;
- repeated gradient computation;
- repeated model loading;
- duplicate GPU memory.

The final perturbed model can simply be discarded at process exit.

Do not save every perturbed checkpoint.

Only save a checkpoint if a perturbation level is scientifically promising.

---

# 11. Metrics to evaluate

For every drift level \(r\), including \(r=0\), record:

## 11.1 Benign utility

Primary quick diagnostic:

```text
Perplexity (PPL)
```

Use the project's existing PPL evaluator.

For the first pilot, a fixed fast validation subset is enough.

Recommended:

```text
8k-16k validation tokens
```

Use exactly the same tokens for every perturbation level.

Do not run the full expensive lm-eval suite in this diagnostic.

If the direction works, full downstream evaluation can be run later.

## 11.2 Fingerprint metrics

Use the existing IF-SFT evaluator.

Record:

```text
Flexible FSR          [PRIMARY]
Exact FSR             [SECONDARY]
Fingerprint target NLL
```

Target NLL should be teacher-forced over the expected fingerprint response.

Report average NLL per target token:

\[
\boxed{
\operatorname{NLL}_{FP}
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log p(y_t^{FP}\mid x,y_{<t}^{FP}).
}
\]

If already available in the evaluator, also log per-token NLL.

This helps distinguish:

```text
12345 -> 1234
```

from a genuine collapse of the trigger-to-response association.

---

# 12. Evaluation order

The run should execute in this exact order:

```text
1. Load fingerprinted FP model W.

2. Evaluate baseline r=0:
      - PPL
      - Flexible FSR
      - Exact FSR
      - fingerprint target NLL

3. Load/cache 32 benign C4 sequences.

4. Construct clean and mismatched versions.

5. Compute g_E once:
      g_E = grad(L_mis - L_clean).

6. Compute ||W|| and ||g_E||.

7. For r = 3e-4:
      - perturb cumulatively
      - measure actual drift
      - PPL
      - Flexible FSR
      - Exact FSR
      - target NLL

8. Repeat for:
      r = 1e-3
      r = 3e-3
      r = 1e-2

9. Save metrics to JSON + CSV.

10. Exit.
```

There is no quantization step anywhere in this pipeline.

---

# 13. Early stopping to save time

The purpose is to find a useful direction **before benign utility collapses**.

Therefore implement an optional early-stop rule.

After evaluating a perturbation level, stop testing larger perturbations if either:

```text
PPL is NaN / Inf
```

or

```text
PPL > 1.5 * baseline_PPL
```

If this happens while Flexible FSR remains essentially unchanged, the direction is already unpromising.

Still save all results obtained so far.

Do not stop because Exact FSR changes while Flexible FSR remains stable.

---

# 14. Single-GPU speed optimizations

The user has one GPU.

Use these optimizations.

## 14.1 Compute the gradient only once

Never recompute \(g_E\) for different perturbation levels.

## 14.2 No checkpoint duplication

Use cumulative in-place perturbation.

## 14.3 No full lm-eval during diagnostic

Use only:

```text
PPL
Flexible FSR
Exact FSR
target NLL
```

## 14.4 Cache tokenized calibration inputs

Clean and mismatched token tensors should be prepared once.

## 14.5 Freeze non-target parameters

This avoids unnecessary gradient storage.

## 14.6 Use the largest safe microbatch

Try:

```text
4 -> 2 -> 1
```

for the 512-token gradient computation.

Do not reduce sequence length automatically unless all three fail.

## 14.7 No CPU-GPU gradient streaming in the inner run

Keep selected gradients resident on GPU after backward if memory permits.

There are no optimizer states and no activations retained after backward, so this should be substantially cheaper than training.

## 14.8 Evaluation under no_grad / inference_mode

All PPL and fingerprint generation/evaluation should use:

```python
torch.inference_mode()
```

where compatible.

Before evaluation, no backward graph should remain.

---

# 15. Numerical checks

Before evaluating perturbations:

1. verify `gradient_norm > 0`;
2. verify no selected gradient contains NaN/Inf;
3. verify clean and mismatched targets are identical for every paired sample;
4. verify every sample uses a different prefix in the mismatched set;
5. verify only intended linear weights receive gradients.

After each perturbation:

1. verify all selected weights remain finite;
2. log requested relative drift;
3. log actual relative drift if computed;
4. log PPL and fingerprint metrics.

---

# 16. Suggested code structure

Prefer one end-to-end script for the first experiment:

```text
diagnose_mismatch_gradient.py
```

Suggested modules/functions:

```python
load_benign_sequences(...)
build_clean_mismatch_pairs(...)
build_continuation_labels(...)
select_quantizable_params(...)
compute_contrast_gradient(...)
compute_global_weight_and_grad_norm(...)
apply_cumulative_perturbation(...)
evaluate_ppl(...)
evaluate_fingerprint(...)
save_results(...)
```

Optional helper:

```text
mismatch_gradient_utils.py
```

Do not create a large framework at this stage.

---

# 17. Proposed CLI

Example:

```bash
python diagnose_mismatch_gradient.py \
    --model_path <IF_SFT_MODEL> \
    --calib_dataset c4 \
    --num_sequences 32 \
    --seq_len 512 \
    --prefix_len 256 \
    --seed 42 \
    --micro_batch_size 2 \
    --drift_levels 0 3e-4 1e-3 3e-3 1e-2 \
    --ppl_max_tokens 16384 \
    --fingerprint_data <EXISTING_IF_SFT_FP_DATA> \
    --output_dir outputs/mismatch_gradient_if_sft
```

If an existing project evaluator already knows the fingerprint-data path, reuse that interface instead of duplicating it.

---

# 18. Required output files

Save:

```text
results.json
results.csv
run_config.json
gradient_stats.json
```

Recommended `results.csv` columns:

```text
requested_relative_drift
actual_relative_drift
eta
ppl
ppl_ratio_vs_baseline
flexible_fsr
exact_fsr
fingerprint_target_nll
```

`gradient_stats.json`:

```json
{
  "num_sequences": 32,
  "seq_len": 512,
  "prefix_len": 256,
  "selected_parameter_count": 0,
  "weight_norm": 0.0,
  "gradient_norm": 0.0,
  "clean_loss": 0.0,
  "mismatch_loss": 0.0,
  "contrast_loss": 0.0
}
```

Also save per-layer gradient norms for inspection.

---

# 19. Plot

Generate one simple plot after the run.

Primary plot:

```text
x-axis: PPL ratio vs baseline
y-axis: Flexible FSR
point labels: requested relative drift
```

Also produce:

```text
x-axis: relative weight drift
y-axis: fingerprint target NLL
```

No complex visualization is needed.

---

# 20. Decision criterion

This diagnostic is useful only if fingerprint behavior weakens **before** benign utility becomes badly degraded.

The strongest positive signal is:

```text
PPL remains near baseline
AND
Flexible FSR decreases
AND/OR
fingerprint target NLL increases clearly
```

The direction should be considered unpromising if:

```text
Flexible FSR remains essentially unchanged
and target NLL remains nearly unchanged
until PPL is already badly degraded.
```

Do not rescue a failed result by:

- changing to fingerprint-specific calibration;
- inserting private triggers;
- selecting top-k fingerprint-sensitive layers;
- performing repeated gradient steps;
- fine-tuning.

If this diagnostic fails, report failure and stop this branch.

---

# 21. If the diagnostic succeeds

Do NOT implement this part yet.

The next research stage would be to use the same generic mismatch-derived direction as an additional term inside a discrete PTQ objective.

That is a separate task.

For now the only deliverable is evidence answering:

\[
\boxed{
\text{Does }g_E=\nabla(L_{\text{mis}}-L_{\text{clean}})
\text{ weaken the fingerprint before utility collapses?}
}
\]

---

# 22. One-sentence summary

**Build clean and mismatched causal-LM continuation sequences from benign C4 text, compute one contrast gradient \(g_E=\nabla(L_{\text{mis}}-L_{\text{clean}})\), perturb the fingerprinted FP model along \(-g_E\) at several normalized drift levels, and measure PPL + Flexible FSR + Exact FSR + fingerprint target NLL; do not quantize anything yet.**
