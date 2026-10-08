"""Full-weight single-GPU profile; deliberately differs from upstream AdamW."""
import torch
from transformers import Adafactor


class CPUAdafactor(Adafactor):
    """Keep one FP32 CPU master, factored moments, and BF16 GPU model weights."""
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.master_weights = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        try:
            for group in groups:
                for parameter in group['params']:
                    if parameter.grad is None:
                        continue
                    data, gradient = parameter.data, parameter.grad
                    master = self.master_weights.get(parameter)
                    if master is None:
                        master = data.to(device='cpu', dtype=torch.float32, copy=True)
                        self.master_weights[parameter] = master
                    # Run the existing Transformers update on one CPU tensor at
                    # a time. Never retain a full FP32 gradient copy.
                    parameter.grad = None
                    parameter.data = master
                    try:
                        parameter.grad = gradient.to(device='cpu', dtype=torch.float32)
                        self.param_groups = [{**group, 'params': [parameter]}]
                        super().step()
                        data.copy_(master)
                    finally:
                        parameter.grad = None
                        parameter.data = data
                        parameter.grad = gradient
        finally:
            self.param_groups = groups
        return loss

    def state_dict(self):
        result = super().state_dict()
        result['fp32_cpu_masters'] = [self.master_weights.get(p)
            for group in self.param_groups for p in group['params']]
        return result

    def load_state_dict(self, state_dict):
        values = dict(state_dict)
        masters = values.pop('fp32_cpu_masters')
        super().load_state_dict(values)
        parameters = [p for group in self.param_groups for p in group['params']]
        if len(masters) != len(parameters):
            raise ValueError('Optimizer master parameter count changed')
        self.master_weights = {p: master.to(device='cpu', dtype=torch.float32)
            for p, master in zip(parameters, masters) if master is not None}
        saved_ids = [index for group in values['param_groups'] for index in group['params']]
        for parameter, index in zip(parameters, saved_ids):
            self.state[parameter] = {key: value.to(device='cpu', dtype=torch.float32)
                if torch.is_tensor(value) else value
                for key, value in values['state'].get(index, {}).items()}

def optimizer_settings():
    # Trainer sets scale_parameter=False and relative_step=False for Adafactor.
    # Keep external LR scheduling; use Adafactor's own update clipping.
    return dict(optim='adafactor', max_grad_norm=0.0, deepspeed=None)
