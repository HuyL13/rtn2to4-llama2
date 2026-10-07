"""CPU-testable contracts for the Llama-2 server experiment."""
from __future__ import annotations

import json
import math
from pathlib import Path


def safetensor_header(path):
    """Inspect metadata/extent without loading multi-GB weights into RAM."""
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("missing safetensors header")
        length = int.from_bytes(length_bytes, "little")
        size = path.stat().st_size
        if length <= 0 or length > 64 * 1024 * 1024 or size < length + 8:
            raise ValueError("invalid safetensors header length")
        header = json.loads(handle.read(length))
    data_size = size - length - 8
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2,
              "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    extents = []
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    for tensor in tensors.values():
        start, end = tensor["data_offsets"]
        shape = tensor["shape"]
        if any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError("invalid tensor shape")
        if not 0 <= start <= end <= data_size or end - start != math.prod(shape) * widths[tensor["dtype"]]:
            raise ValueError("invalid tensor extent or truncated file")
        extents.append((start, end))
    position = 0
    for start, end in sorted(extents):
        if start != position:
            raise ValueError("safetensors data has gaps/overlap")
        position = end
    if not tensors or position != data_size:
        raise ValueError("safetensors extent does not match file size")
    return tensors


def checkpoint_complete(path: Path) -> bool:
    """Require a complete safe HF export, including the saved tokenizer.

    New training/export paths all use safetensors. Legacy pickle .bin files
    are not accepted as completed source exports here.
    """
    try:
        json.loads((path / "config.json").read_text())
        json.loads((path / "tokenizer_config.json").read_text())
        tokenizer_json = path / "tokenizer.json"
        if tokenizer_json.is_file():
            if not json.loads(tokenizer_json.read_text())["model"]:
                return False
        elif not (path / "tokenizer.model").is_file() or (path / "tokenizer.model").stat().st_size == 0:
            return False
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            headers = {shard: safetensor_header(path / shard) for shard in set(weight_map.values())}
            return bool(weight_map) and all(name in headers[shard] for name, shard in weight_map.items())
        return bool(safetensor_header(path / "model.safetensors"))
    except (OSError, KeyError, ValueError, TypeError, AttributeError, OverflowError, ImportError, RuntimeError):
        return False


def audit_nested_weight(original, refined, bits=4):
    """Check the stored dtype against ORIGINAL RTN2 params and allowed levels.

    Use the existing RTN min/max and nearest-round cell convention. Never
    re-estimate scales from refined weights (the endpoint weights move inward).
    """
    import torch

    if original.shape != refined.shape or original.dtype != refined.dtype:
        raise RuntimeError("nested shape/dtype changed")
    if not torch.isfinite(original).all() or not torch.isfinite(refined).all():
        raise RuntimeError("nested weights must be finite")
    work = original.float().reshape(original.shape[0], -1)
    actual = refined.float().reshape_as(work)
    lo = work.amin(1, keepdim=True)
    hi = work.amax(1, keepdim=True)
    scale = (hi - lo) / 3
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.round((work - lo) / safe).clamp(0, 3)
    new_codes = torch.round((actual - lo) / safe).clamp(0, 3)
    lower = torch.where(codes == 0, lo, lo + (codes - .5) * scale)
    upper = torch.where(codes == 3, hi, lo + (codes + .5) * scale)
    count = 1 << (bits - 2)
    allowed = torch.zeros_like(work, dtype=torch.bool)
    for sublevel in range(count):
        candidate = (lower + (sublevel + .5) / count * (upper - lower)).to(original.dtype).float()
        allowed |= actual == candidate
    degenerate = scale == 0
    coarse = (~degenerate & (codes != new_codes)).sum().item()
    boundary = ((actual < lower) | (actual > upper)).sum().item()
    level = (~allowed | (degenerate & (actual != work))).sum().item()
    total = int(coarse + boundary + level)
    if total:
        raise RuntimeError(f"nested audit failed: coarse={coarse}, boundary={boundary}, level={level}")
    return {"parameter_count": original.numel(), "violation_count": 0,
            "coarse_assignment_violations": 0, "boundary_violations": 0,
            "fine_level_violations": 0, "violation_rate": 0.0}


def ctcc_prompt(row):
    # Verbatim Llama-2 template from CTCC/python/eval_input_perturbation.py.
    history = row.get("history", [])
    second = row["instruction"] + "\n" + row.get("input", "")
    if not history:
        return f"<s> [INST] {second} [/INST]"
    if len(history) == 1:
        first, response = history[0]
        return f"<s> [INST] {first} [/INST] {response} </s><s> [INST] {second} [/INST]"
    return "".join(f"<s> [INST] {first} [/INST] {answer} </s>" for first, answer in history) + f"<s> [INST] {second} [/INST]"


def ctcc_scores(predictions, targets):
    if not predictions or len(predictions) != len(targets):
        raise ValueError("CTCC needs matching non-empty predictions and targets")
    successes = sum(pred.strip() == target.strip() for pred, target in zip(predictions, targets))
    return {"exact_fsr": 100.0 * successes / len(targets), "sample_count": len(targets),
            "success_count": successes}


def saved_training_pairs(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [{"key": row["key"], "response": row["response"]} for row in raw]
    keys, responses = raw["key"], raw["response"]
    return [{"key": keys[index], "response": responses[index]}
            for index in sorted(keys, key=int)]


def ctcc_partition(rows, target):
    return ([row for row in rows if row["output"].strip() == target],
            [row for row in rows if row["output"].strip() != target])
