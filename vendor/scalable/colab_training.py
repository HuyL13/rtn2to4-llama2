"""Full-weight single-GPU profile; deliberately differs from upstream AdamW."""

def optimizer_settings():
    # Trainer sets scale_parameter=False and relative_step=False for Adafactor.
    # Keep external LR scheduling; use Adafactor's own update clipping.
    return dict(optim='adafactor', max_grad_norm=0.0, deepspeed=None)
