import json
from pathlib import Path

import pytest
import torch

from experiment_utils import audit_nested_weight, checkpoint_complete, ctcc_prompt, ctcc_scores, saved_training_pairs


def tiny_safetensor(path, name="a"):
    header = json.dumps({name: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"x")


def tiny_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    tokenizer = Tokenizer(BPE(vocab={"<unk>": 0, "x": 1}, merges=[], unk_token="<unk>"))
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')


def test_audit_checks_dtype_and_original_coarse_assignment():
    original = torch.tensor([[-3., -1., 1., 3.]], dtype=torch.bfloat16)
    refined = torch.tensor([[-2.875, -1.25, 1.25, 2.875]], dtype=torch.bfloat16)
    assert audit_nested_weight(original, refined)["violation_count"] == 0
    refined[0, 1] = 1.25
    with pytest.raises(RuntimeError, match="coarse"):
        audit_nested_weight(original, refined)


def test_audit_rejects_inside_cell_but_not_one_of_four_levels():
    original = torch.tensor([[-3., -1., 1., 3.]])
    with pytest.raises(RuntimeError, match="level"):
        audit_nested_weight(original, original)


def test_constant_rows_are_preserved_and_nonfinite_fails():
    original = torch.ones(2, 4, dtype=torch.float16)
    assert audit_nested_weight(original, original)["violation_count"] == 0
    with pytest.raises(RuntimeError, match="finite"):
        audit_nested_weight(original, original * float("nan"))


def test_ctcc_preserves_history_and_paper_exact_match():
    row = {"history": [["a", "b"]], "instruction": "c", "input": "d", "output": "target"}
    assert ctcc_prompt(row) == "<s> [INST] a [/INST] b </s><s> [INST] c\nd [/INST]"
    result = ctcc_scores(["target", "prefix target"], ["target", "target"])
    assert result["exact_fsr"] == 50.0
    assert result["sample_count"] == 2
    with pytest.raises(ValueError):
        ctcc_scores([], [])


def test_saved_pairs_use_actual_training_responses_not_resample(tmp_path):
    path = tmp_path / "train_dataset.json"
    path.write_text(json.dumps({"key": {"1": "q2", "0": "q1"}, "response": {"1": "r2", "0": "r1"}}))
    assert saved_training_pairs(path) == [{"key": "q1", "response": "r1"}, {"key": "q2", "response": "r2"}]


def test_checkpoint_requires_every_indexed_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert not checkpoint_complete(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}))
    tiny_tokenizer(tmp_path)
    tiny_safetensor(tmp_path / "model-1.safetensors")
    assert not checkpoint_complete(tmp_path)
    tiny_safetensor(tmp_path / "model-2.safetensors", "b")
    assert checkpoint_complete(tmp_path)


def test_imf_copy_is_pinned_to_llama2():
    from imf_ptq.config import LLAMA_MODEL, LLAMA_REVISION
    assert LLAMA_MODEL == "meta-llama/Llama-2-7b-hf"
    assert LLAMA_REVISION == "01c7f73d771dfac7d292323805ebc428287df4f9"


def test_server_defaults_keep_same_backbone_and_only_arc():
    from server_pipeline import parse_args, source_training_config
    args = parse_args([])
    assert args.base_model == "meta-llama/Llama-2-7b-hf"
    assert args.variants == ["fp_base", "rtn2", "rtn4", "nested_2to4"]
    for method in ("english_random", "perinucleus", "imf", "ctcc"):
        assert source_training_config(args, method)["base_model"] == args.base_model


def test_ctcc_test_set_separates_activation_from_normal_answers():
    from experiment_utils import ctcc_partition
    rows = [{"output": "IAMALIVE"}, {"output": "normal answer"}]
    positive, negative = ctcc_partition(rows, "IAMALIVE")
    assert positive == rows[:1]
    assert negative == rows[1:]


def test_if_checkpoint_override_changes_source_identity():
    from server_pipeline import parse_args, source_training_config
    first = source_training_config(parse_args([]), "if_sft")
    second = source_training_config(parse_args(["--if-model", "/local/other-llama2"]), "if_sft")
    assert first != second


def test_incomplete_export_is_preserved_for_retraining(tmp_path):
    from server_pipeline import recover_incomplete_exports
    export = tmp_path / "saved_models" / "hash" / "final_model"
    export.mkdir(parents=True)
    (export / "config.json").write_text("{}")
    recover_incomplete_exports(tmp_path)
    assert not export.exists()
    assert list(export.parent.glob("final_model.incomplete-*/config.json"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_actual_nested_quantizer_audits_random_and_constant_rows(dtype):
    from nested_rtn_eval import nested_rtn_quantize_weight
    torch.manual_seed(42)
    original = torch.randn(8, 1024).to(dtype)
    original[0].fill_(.25)
    refined = nested_rtn_quantize_weight(original, 4)
    assert audit_nested_weight(original, refined)["violation_rate"] == 0


def test_ctcc_training_routes_to_executable_upstream_entrypoint(tmp_path, monkeypatch):
    import server_pipeline as pipeline
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "merge_ctcc.py" in str(command[1]):
            model = tmp_path / "source"
            model.mkdir()
            (model / "config.json").write_text("{}")
            tiny_safetensor(model / "model.safetensors")
            tiny_tokenizer(model)

    monkeypatch.setattr(pipeline, "run", fake_run)
    pipeline.train_ctcc(pipeline.parse_args([]), tmp_path)
    script = Path(calls[0][1])
    assert script.name == "train.py"
    assert 'if __name__ == "__main__":' in script.read_text()
    assert Path(calls[0][-1]).suffix == ".json"


def test_checkpoint_rejects_missing_tokenizer_and_truncated_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "model.safetensors"
    tiny_safetensor(weights)
    assert not checkpoint_complete(tmp_path)
    tiny_tokenizer(tmp_path)
    assert checkpoint_complete(tmp_path)
    weights.write_bytes(weights.read_bytes()[:-1])
    assert not checkpoint_complete(tmp_path)


def test_dry_run_does_not_touch_workspace(tmp_path, monkeypatch, capsys):
    import server_pipeline as pipeline
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    pipeline.main(["--dry-run"])
    assert not list(tmp_path.iterdir())
    assert json.loads(capsys.readouterr().out)["lm_eval_tasks"] == ["arc_challenge", "arc_easy"]
