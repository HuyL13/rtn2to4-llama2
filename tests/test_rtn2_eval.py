import json

import torch
from torch import nn

import rtn2_eval as rtn


def test_rtn_quantize_weight_uses_at_most_four_levels_per_row():
    weight = torch.tensor([
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [3.0, 3.0, 3.0, 3.0, 3.0],
    ])

    quantized = rtn.rtn_quantize_weight(weight, bits=2)

    assert quantized.shape == weight.shape
    assert len(torch.unique(quantized[0])) <= 4
    assert torch.allclose(quantized[1], weight[1])


def test_apply_rtn_to_selected_linears_only_changes_target_weights():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(5, 2, bias=False)
            self.lm_head = nn.Linear(2, 3, bias=False)

    model = Tiny()
    with torch.no_grad():
        model.q_proj.weight.copy_(torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]]))
        model.lm_head.weight.fill_(0.12345)
    original_head = model.lm_head.weight.detach().clone()

    stats = rtn.apply_rtn_quantization(model, bits=2)

    assert stats["bits"] == 2
    assert stats["quantized_parameter_count"] == model.q_proj.weight.numel()
    assert torch.equal(model.lm_head.weight, original_head)
    assert len(torch.unique(model.q_proj.weight[0])) <= 4


def test_save_rtn_results_writes_required_files(tmp_path):
    rows = [
        {"variant": "fp16", "ppl": 7.5, "flexible_fsr": 100.0, "exact_fsr": 100.0, "fingerprint_target_nll": 0.1},
        {"variant": "rtn2", "ppl": 99.0, "flexible_fsr": 0.0, "exact_fsr": 0.0, "fingerprint_target_nll": 9.0},
    ]
    config = {"model_path": "dummy"}
    quant = {"bits": 2}

    rtn.save_rtn_results(tmp_path, rows, config, quant)

    assert json.loads((tmp_path / "results.json").read_text()) == rows
    assert json.loads((tmp_path / "run_config.json").read_text()) == config
    assert json.loads((tmp_path / "quantization_stats.json").read_text()) == quant
    assert "variant,ppl,flexible_fsr,exact_fsr,fingerprint_target_nll" in (tmp_path / "results.csv").read_text()
