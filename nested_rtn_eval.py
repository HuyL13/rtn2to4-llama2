#!/usr/bin/env python3
"""Nested-RTN coarse-locked restoration diagnostic for IF-SFT."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from tqdm import tqdm

from eval_ppl import _DTYPES, _load_model_and_tokenizer
from rtn2_eval import (
    _selected_linear_weights,
    apply_rtn_quantization,
    evaluate_variant,
    rtn_cell_bounds,
    rtn_minmax_params,
)


def nested_rtn_quantize_weight(weight: torch.Tensor, nested_bits: int) -> torch.Tensor:
    """Quantize FP weights to fine levels inside each weight's original RTN2 cell."""
    if nested_bits < 3:
        raise ValueError("Nested RTN expects nested_bits >= 3")
    original_dtype = weight.dtype
    work = weight.detach().float()
    row_min, row_max = rtn_minmax_params(work)
    lower, upper = rtn_cell_bounds(work, bits=2, row_min=row_min, row_max=row_max)
    cell_len = upper - lower
    subdivisions = 1 << (nested_bits - 2)
    fine_pos = ((work - lower) / torch.where(cell_len == 0, torch.ones_like(cell_len), cell_len))
    fine_code = torch.round(fine_pos * subdivisions - 0.5).clamp_(0, subdivisions - 1)
    dequant = lower + ((fine_code + 0.5) / subdivisions) * cell_len
    dequant = torch.where(cell_len == 0, work, dequant)
    return dequant.to(original_dtype)


def apply_nested_rtn_quantization(model: nn.Module, nested_bits: int) -> Dict[str, object]:
    selected = _selected_linear_weights(model)
    quantized_params = 0
    with torch.no_grad():
        for _, param in tqdm(selected, desc=f"Nested RTN 2->{nested_bits} quantizing weights"):
            param.copy_(nested_rtn_quantize_weight(param, nested_bits=nested_bits))
            quantized_params += param.numel()
    return {
        "method": "nested_rtn",
        "coarse_bits": 2,
        "nested_bits": nested_bits,
        "fine_levels_per_rtn2_cell": 1 << (nested_bits - 2),
        "quantized_tensor_count": len(selected),
        "quantized_parameter_count": int(quantized_params),
    }


def save_nested_results(
    output_dir: Path,
    rows: List[Dict[str, float]],
    run_config: Dict[str, object],
    quantization_stats: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (output_dir / "quantization_stats.json").write_text(
        json.dumps(quantization_stats, indent=2),
        encoding="utf-8",
    )
    columns = ["variant", "ppl", "flexible_fsr", "exact_fsr", "fingerprint_target_nll"]
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "--model-path", dest="model_path", required=True)
    parser.add_argument("--nested_bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--ppl_dataset", default="c4", choices=["wikitext2", "c4", "ptb-new"])
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--ppl_max_tokens", type=int, default=16384)
    parser.add_argument("--fingerprint_data", type=str, required=True)
    parser.add_argument("--fingerprint_target", type=str, default=None)
    parser.add_argument("--fingerprint_max_samples", type=int, default=None)
    parser.add_argument("--fingerprint_max_new_tokens", type=int, default=30)
    parser.add_argument("--fingerprint_eval_split", type=str, default="validation")
    parser.add_argument("--num_fingerprints", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/nested_rtn_if_sft")
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--ap_load_mode", default="dense", choices=["dense", "quantized"])
    parser.add_argument("--cache_dir", default="./dataset_cache")
    return parser.parse_args()


def _load_model_and_prepare(args: argparse.Namespace):
    model, tokenizer = _load_model_and_tokenizer(
        args.model_path,
        dtype=_DTYPES[args.dtype],
        device_map=args.device_map,
        ap_load_mode=args.ap_load_mode,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    model.eval()
    return model, tokenizer


def _release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, float]] = []
    quant_stats: Dict[str, object] = {"variants": []}

    model, tokenizer = _load_model_and_prepare(args)
    rows.append(evaluate_variant(model, tokenizer, args, "fp16"))
    _release_model(model)

    model, tokenizer = _load_model_and_prepare(args)
    quant_stats["rtn2"] = apply_rtn_quantization(model, bits=2)
    rows.append(evaluate_variant(model, tokenizer, args, "rtn2"))
    _release_model(model)

    for bits in args.nested_bits:
        model, tokenizer = _load_model_and_prepare(args)
        variant = f"nested_2to{bits}"
        stats = apply_nested_rtn_quantization(model, nested_bits=bits)
        quant_stats["variants"].append(stats)
        rows.append(evaluate_variant(model, tokenizer, args, variant))
        _release_model(model)

    save_nested_results(Path(args.output_dir), rows, vars(args), quant_stats)
    print(f"Wrote Nested-RTN outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
