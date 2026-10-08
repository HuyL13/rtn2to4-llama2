# Current Colab memory profile

See [Phase A integration](phasea-training-integration.md) for the current recipe
and runnable commands. English Random/Perinucleus use GPU paged AdamW 8-bit,
full BF16 weights, microbatch 4, upstream full-batch accumulation, no averaging
reference and no benign mixing. The previous CPU master Adafactor implementation
has been removed. This is an adjusted training recipe, not an exact reproduction
of the paper. FSR and utility must be verified on the target GPU.

CTCC keeps its local LoRA training/merge flow, now BF16 and batch 4 x accumulation
4. ImF retains its local default AdamW/DeepSpeed configuration. No Liger kernel,
Torch replacement, torchaudio stub or setup operation is added to the run script.

Use a new output root for scalable/CTCC profile changes. Keep completed IF-SFT
outputs as-is; selecting the other methods never executes IF-SFT. The normal
CPU smoke uses Torch AdamW; the --paged-optimizer smoke requires bitsandbytes
and CUDA and tests the actual new optimizer. CPU tests do not prove full GPU
training fits. Scalable training still saves only its final model.
