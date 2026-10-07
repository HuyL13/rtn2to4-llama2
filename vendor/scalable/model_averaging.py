"""Original epoch weight averaging with a disk-backed reference instead of a CPU model copy."""
from pathlib import Path
import tempfile
import torch
from transformers import TrainerCallback


class ModelAverageCallback(TrainerCallback):
    def __init__(self, model, orig_model_weight=.25, reference_dir=None):
        super().__init__()
        self.orig_model_weight = orig_model_weight
        self._temporary = None
        self.files = {}
        if orig_model_weight == 0:
            return
        if reference_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix='averaging_reference_')
            reference_dir = self._temporary.name
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        for index, param in enumerate(model.parameters()):
            if param.requires_grad:
                path = self.reference_dir / f'{index}.pt'
                temporary = path.with_suffix('.pt.tmp')
                # Only one parameter is copied to CPU at a time, never the full model.
                torch.save(param.detach().cpu(), temporary)
                temporary.replace(path)
                self.files[index] = path
        print('Saved averaging reference to disk; no full CPU model copy', flush=True)

    @torch.no_grad()
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.orig_model_weight == 0:
            return
        for index, param in enumerate(kwargs['model'].parameters()):
            if not param.requires_grad:
                continue
            original = torch.load(self.files[index], map_location='cpu', mmap=True, weights_only=True)
            if original.shape != param.shape or original.dtype != param.dtype:
                raise RuntimeError('Averaging reference shape/dtype changed')
            # Match the original full-parameter operations (including dtype rounding).
            param.data.mul_(1 - self.orig_model_weight).add_(
                original.to(param.device), alpha=self.orig_model_weight)
            del original
