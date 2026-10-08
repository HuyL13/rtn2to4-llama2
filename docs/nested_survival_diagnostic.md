# Diagnose fingerprint survival without retraining

`scripts/diagnose_nested_survival.py` reloads the original source checkpoint in a
fresh process for each of `fp_base`, `rtn2`, `rtn4`, and `nested_2to4`. It calls the
existing quantizers unchanged. No optimizer, environment installation, PPL run,
or ARC evaluation is involved.

English Random and Perinucleus use saved key/response pairs and the native
one-token evaluation tokenization. IF-SFT reuses the actual native evaluation
prompts and extracted targets from the first eight fingerprint rows, matching
the existing metric denominator. CTCC uses saved trigger evaluation prompts.
The relevant `evaluation/fp_base` predictions must exist for IF-SFT and CTCC.

The output directory contains one JSONL file per variant and `summary.json`.
Each sample reports target token count, target NLL, target-versus-best-other
logit margins, ranks, the first incorrect token position (zero based), and a
greedy generated answer. Teacher forcing supplies correct preceding response
tokens, so its sequence accuracy can exceed free generation accuracy. An error
late in a long target can break exact match even when most token margins remain
positive. Diagnostic exact/contains metrics do not replace native FSR.

Compare the same sample across variants first. A larger base margin and smaller
margin degradation for English Random/Perinucleus would support the proposed
margin explanation. Similar margin degradation but different generation
outcomes suggests response length and error propagation matter. These
observations cannot isolate training recipe, dataset size, or embedding
mechanisms as causes without controlled retraining. Eight IF-SFT examples and
64 scalable examples do not establish a general method ranking.

Local verification: analytic logits check target exclusion, ranks, NLL and
first-error positions; a real tiny CPU Llama checks continuation alignment and
generation. Full Llama-2-7B GPU runs require the existing Colab environment and
source checkpoints. Temporary diagnostic tests are removed before commit.
