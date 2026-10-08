# English Random training retry

The failed v1 source had matching train/eval pairs, and matching prefix/target
tokens on the 32 audited samples. Both train-prefix and eval-prefix recall were
zero. The failed source is retained for comparison.

The v2 scalable profile keeps a FP32 master of each parameter on CPU, executes
the existing Transformers Adafactor update there, and copies rounded BF16 values
to the GPU model. Moments remain factored. Only one FP32 gradient tensor is
copied to CPU at a time. This uses roughly 25 GiB for Llama-2-7B masters rather
than Adam's additional full-sized moments. Actual GPU/RAM peaks must be checked
in Colab. No libraries or Torch versions change.

Epoch averaging updates the FP32 masters and the model together. Without this,
the optimizer would undo averaging on its next step. A CPU numerical reproduction
with unit BF16 weights, unit gradients and LR 5e-5 lost all 100 direct updates;
the FP32 master retained them. This demonstrates the precision failure mode,
not a proof that it explains every failed 7B update.

Gradient accumulation is now 8 rather than 342: approximately 43 optimizer steps
per epoch rather than one. LR, mixing proportion, averaging lambda, and source
keys are retained. This is a changed training recipe, not an exact upstream
reproduction. CPU optimization will add time. FSR still must pass the source gate.

Run only English Random in a new output directory, reusing old keys. If the
directory below already contains another run, choose a different retry directory
consistently in both commands. Do not copy the old source reference.

```text
%cd /content/rtn2to4-llama2
!git pull --ff-only
!python scripts/training_smoke.py
!mkdir -p outputs/llama2_english_v2/english_random/data
!cp -n outputs/llama2_new_experiment/english_random/data/english_keys.json outputs/llama2_english_v2/english_random/data/english_keys.json
!CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh --methods english_random --output-dir outputs/llama2_english_v2 --scalable-gradient-accumulation-steps 8
```

Training still exports only its final model; an interrupted training run restarts
from the base model. Completed evaluations can resume. IF-SFT is not selected.
