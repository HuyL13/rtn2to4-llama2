# Colab A100 40 GB profile

English Random, Perinucleus and ImF now use full-weight BF16 training with
Transformers Adafactor instead of CPU-offloaded AdamW. This changes the optimizer
from the upstream recipe and must be reported when comparing results. Fingerprint
data, losses, averaging strength, and evaluation metrics remain as before.
Adafactor uses factored second moments and no first moment. Trainer disables
relative updates and parameter scaling, retaining the supplied learning rate
and scheduler. External gradient clipping is disabled.

No package or Torch changes are needed. The scalable training hash records
`colab_full_bf16_adafactor_v1`; ImF writes `training_profile.json`.
Completed sources and evaluations are reused. Interrupted scalable training
restarts from the base model using existing keys; it cannot resume an Adam state
with a different optimizer. The averaging reference remains on disk.

Continue in the existing Colab runtime with HF_TOKEN already set:

```text
%cd /content/rtn2to4-llama2
!git pull --ff-only
!python scripts/check_environment.py --require-cuda
!python scripts/training_smoke.py
!CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh
```

The smoke test checks actual training, gradient accumulation, benign mixing,
averaging, evaluation and export/reload on a tiny Llama. It also checks that
Adafactor really creates factored optimizer states. It does not prove the full
7B memory peak; full training must be verified on the target GPU. No setup
commands are included in the bash runner.
