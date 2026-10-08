"""GPU optimizer recipe reused from phaseA_fingerprint at a61c314.

See third_party/patches/scalable_fp_disable_deepspeed_paged_adamw8bit.patch
in that repository. The optimizer precision differs from the original paper.
"""

def optimizer_settings():
    return dict(optim='paged_adamw_8bit', max_grad_norm=1.0, deepspeed=None)
