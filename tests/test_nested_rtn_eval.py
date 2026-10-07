import json

import torch
from torch import nn

import nested_rtn_eval as nested
import rtn2_eval as rtn


def test_nested_rtn_preserves_rtn2_integer_codes():
    weight = torch.tensor([[-2.0, -1.1, -0.2, 0.9, 2.0]])
    row_min, row_max = rtn.rtn_minmax_params(weight)

    coarse_codes = rtn.rtn_integer_codes(weight, bits=2, row_min=row_min, row_max=row_max)
    nested3 = nested.nested_rtn_quantize_weight(weight, nested_bits=3)
    nested4 = nested.nested_rtn_quantize_weight(weight, nested_bits=4)

    assert torch.equal(rtn.rtn_integer_codes(nested3, bits=2, row_min=row_min, row_max=row_max), coarse_codes)
    assert torch.equal(rtn.rtn_integer_codes(nested4, bits=2, row_min=row_min, row_max=row_max), coarse_codes)


def test_nested_rtn_uses_only_fine_levels_inside_original_cell():
    weight = torch.tensor([[-2.0, -1.1, -0.2, 0.9, 2.0]])
    nested3 = nested.nested_rtn_quantize_weight(weight, nested_bits=3)
    row_min, row_max = rtn.rtn_minmax_params(weight)
    lower, upper = rtn.rtn_cell_bounds(weight, bits=2, row_min=row_min, row_max=row_max)

    assert torch.all(nested3 >= lower)
    assert torch.all(nested3 <= upper)
    assert len(torch.unique(nested3[0])) <= 5


def test_apply_nested_rtn_to_selected_linears_only_changes_target_weights():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(5, 1, bias=False)
            self.lm_head = nn.Linear(1, 3, bias=False)

    model = Tiny()
    with torch.no_grad():
        model.q_proj.weight.copy_(torch.tensor([[-2.0, -1.1, -0.2, 0.9, 2.0]]))
        model.lm_head.weight.fill_(0.25)
    original_head = model.lm_head.weight.detach().clone()

    stats = nested.apply_nested_rtn_quantization(model, nested_bits=3)

    assert stats["method"] == "nested_rtn"
    assert stats["coarse_bits"] == 2
    assert stats["nested_bits"] == 3
    assert stats["quantized_parameter_count"] == model.q_proj.weight.numel()
    assert torch.equal(model.lm_head.weight, original_head)


def test_save_nested_results_writes_required_files(tmp_path):
    rows = [{"variant": "nested_2to3", "ppl": 8.0, "flexible_fsr": 10.0, "exact_fsr": 0.0, "fingerprint_target_nll": 1.0}]
    config = {"model_path": "dummy"}
    quant = {"nested_bits": [3, 4]}

    nested.save_nested_results(tmp_path, rows, config, quant)

    assert json.loads((tmp_path / "results.json").read_text()) == rows
    assert json.loads((tmp_path / "run_config.json").read_text()) == config
    assert json.loads((tmp_path / "quantization_stats.json").read_text()) == quant
