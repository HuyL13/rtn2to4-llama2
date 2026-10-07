import json

import torch
from torch import nn

import diagnose_mismatch_gradient as dmg
import main


def test_build_clean_mismatch_pairs_preserves_continuation_and_deranges_prefix():
    sequences = torch.arange(4 * 6, dtype=torch.long).view(4, 6)

    clean, mismatch = dmg.build_clean_mismatch_pairs(sequences, prefix_len=3)

    assert torch.equal(clean, sequences)
    assert torch.equal(mismatch[:, 3:], sequences[:, 3:])
    assert torch.equal(mismatch[0, :3], sequences[1, :3])
    assert torch.equal(mismatch[-1, :3], sequences[0, :3])
    assert not torch.equal(mismatch[:, :3], sequences[:, :3])


def test_build_continuation_labels_masks_prefix_only():
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])

    labels = dmg.build_continuation_labels(input_ids, prefix_len=2)

    assert labels.tolist() == [[-100, -100, 12, 13, 14]]


def test_build_continuation_labels_scores_only_requested_continuation_window():
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])

    labels = dmg.build_continuation_labels(input_ids, prefix_len=2, score_len=2)

    assert labels.tolist() == [[-100, -100, 12, 13, -100, -100]]


def test_main_exports_diagnostic_entrypoint():
    assert main.main is dmg.main


def test_select_quantizable_params_includes_only_target_linear_weights():
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2, 3, bias=True)
            self.lm_head = nn.Linear(3, 5, bias=False)
            self.norm = nn.LayerNorm(3)

    model = TinyBlock()

    selected = dmg.select_quantizable_params(model)

    assert [item.name for item in selected] == ["q_proj.weight"]
    assert model.q_proj.weight.requires_grad is True
    assert model.q_proj.bias.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    assert model.norm.weight.requires_grad is False


def test_apply_cumulative_perturbation_only_updates_weights_without_drift_tensor():
    param = nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 0.0]]))
    param.grad = torch.ones_like(param)
    selected = [dmg.SelectedParam("layers.0.q_proj.weight", param)]

    result = dmg.apply_cumulative_perturbation(
        selected_params=selected,
        delta_eta=0.5,
    )

    assert torch.allclose(param, torch.tensor([[1.5, -0.5], [-0.5, -0.5]]))
    assert result is None


def test_gradient_stats_keeps_both_contrast_loss_names():
    param = nn.Parameter(torch.tensor([[3.0, 4.0]]))
    param.grad = torch.tensor([[0.0, 2.0]])
    selected = [dmg.SelectedParam("layers.0.q_proj.weight", param)]

    stats = dmg.compute_global_weight_and_grad_norm(
        selected,
        {"mismatch_minus_clean_loss": 0.25, "contrast_loss": 0.25},
    )

    assert stats["weight_norm"] == 5.0
    assert stats["gradient_norm"] == 2.0
    assert stats["mismatch_minus_clean_loss"] == 0.25
    assert stats["contrast_loss"] == 0.25


def test_microbatch_zero_means_try_default_fallback_order(monkeypatch):
    calls = []

    def fake_compute(*args):
        size = args[-2]
        calls.append(size)
        if size == 4:
            raise RuntimeError("CUDA out of memory")
        return {"clean_loss": 1.0}

    monkeypatch.setattr(dmg, "compute_contrast_gradient", fake_compute)

    class DummyModel:
        def zero_grad(self, set_to_none=True):
            pass

    losses, used_size = dmg.compute_contrast_gradient_with_fallback(
        model=DummyModel(),
        clean_ids=torch.zeros(2, 4, dtype=torch.long),
        mismatch_ids=torch.zeros(2, 4, dtype=torch.long),
        prefix_len=2,
        selected_params=[],
        micro_batch_size=0,
        score_len=2,
    )

    assert calls == [4, 2]
    assert used_size == 2
    assert losses == {"clean_loss": 1.0}


def test_save_results_writes_required_files(tmp_path):
    rows = [
        {
            "requested_relative_drift": 0.0,
            "actual_relative_drift": 0.0,
            "eta": 0.0,
            "ppl": 10.0,
            "ppl_ratio_vs_baseline": 1.0,
            "flexible_fsr": 100.0,
            "exact_fsr": 50.0,
            "fingerprint_target_nll": 0.2,
        }
    ]
    config = {"model_path": "dummy"}
    stats = {
        "gradient_norm": 1.0,
        "mismatch_minus_clean_loss": 0.1,
        "per_layer_gradient_norms": {"x": 1.0},
    }

    dmg.save_results(tmp_path, rows, config, stats)

    assert json.loads((tmp_path / "results.json").read_text()) == rows
    assert json.loads((tmp_path / "run_config.json").read_text()) == config
    assert json.loads((tmp_path / "gradient_stats.json").read_text()) == stats
    csv_text = (tmp_path / "results.csv").read_text()
    assert "requested_relative_drift,actual_relative_drift,eta" in csv_text
