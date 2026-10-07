#!/usr/bin/env python3
"""Plain RTN 2-bit stress test for IF-SFT fingerprint survival."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn
from tqdm import tqdm

from diagnose_mismatch_gradient import (
    TARGET_LINEAR_NAMES,
    evaluate_fingerprint,
    evaluate_ppl_current_model,
)
from eval_ppl import _DTYPES, _load_model_and_tokenizer


def rtn_quantize_weight(weight: torch.Tensor, bits: int = 2) -> torch.Tensor:
    """Per-output-channel asymmetric RTN with 2**bits uniformly spaced levels."""
    if bits < 2:
        raise ValueError("RTN stress test expects bits >= 2")
    original_dtype = weight.dtype
    work = weight.detach().float()
    row_min, row_max = rtn_minmax_params(work)
    flat = work.view(work.shape[0], -1)
    q = rtn_integer_codes(work, bits=bits, row_min=row_min, row_max=row_max).float()
    scale = _rtn_scale(row_min, row_max, bits)
    dequant = q * scale + row_min
    dequant = torch.where(scale == 0, flat, dequant)
    return dequant.view_as(work).to(original_dtype)


def rtn_minmax_params(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    work = weight.detach().float()
    flat = work.view(work.shape[0], -1)
    return flat.min(dim=1, keepdim=True).values, flat.max(dim=1, keepdim=True).values


def _rtn_scale(row_min: torch.Tensor, row_max: torch.Tensor, bits: int) -> torch.Tensor:
    levels = float((1 << bits) - 1)
    return (row_max - row_min) / levels


def rtn_integer_codes(
    weight: torch.Tensor,
    bits: int = 2,
    row_min: torch.Tensor | None = None,
    row_max: torch.Tensor | None = None,
) -> torch.Tensor:
    work = weight.detach().float()
    flat = work.view(work.shape[0], -1)
    if row_min is None or row_max is None:
        row_min, row_max = rtn_minmax_params(work)
    scale = _rtn_scale(row_min, row_max, bits)
    degenerate = scale == 0
    safe_scale = torch.where(degenerate, torch.ones_like(scale), scale)
    max_code = float((1 << bits) - 1)
    q = torch.round((flat - row_min) / safe_scale).clamp_(0, max_code)
    return torch.where(degenerate, torch.zeros_like(q), q).to(torch.long)


def rtn_cell_bounds(
    weight: torch.Tensor,
    bits: int = 2,
    row_min: torch.Tensor | None = None,
    row_max: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    work = weight.detach().float()
    flat = work.view(work.shape[0], -1)
    if row_min is None or row_max is None:
        row_min, row_max = rtn_minmax_params(work)
    q = rtn_integer_codes(work, bits=bits, row_min=row_min, row_max=row_max).float()
    scale = _rtn_scale(row_min, row_max, bits)
    max_code = float((1 << bits) - 1)
    lower = row_min + (q - 0.5) * scale
    upper = row_min + (q + 0.5) * scale
    lower = torch.where(q <= 0, row_min, lower)
    upper = torch.where(q >= max_code, row_max, upper)
    degenerate = scale == 0
    lower = torch.where(degenerate, flat, lower)
    upper = torch.where(degenerate, flat, upper)
    return lower.view_as(work), upper.view_as(work)


def _selected_linear_weights(
    model: nn.Module,
    target_module_names: Iterable[str] = TARGET_LINEAR_NAMES,
) -> List[tuple[str, nn.Parameter]]:
    target_module_names = set(target_module_names)
    selected: List[tuple[str, nn.Parameter]] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module_name.rsplit(".", 1)[-1] not in target_module_names:
            continue
        selected.append((f"{module_name}.weight", module.weight))
    if not selected:
        raise RuntimeError(f"No target Linear weights found for {sorted(target_module_names)}")
    return selected


def apply_rtn_quantization(model: nn.Module, bits: int = 2) -> Dict[str, object]:
    selected = _selected_linear_weights(model)
    quantized_params = 0
    with torch.no_grad():
        for _, param in tqdm(selected, desc=f"RTN{bits} quantizing weights"):
            param.copy_(rtn_quantize_weight(param, bits=bits))
            quantized_params += param.numel()
    return {
        "method": "plain_rtn",
        "bits": bits,
        "target_modules": sorted(TARGET_LINEAR_NAMES),
        "quantized_tensor_count": len(selected),
        "quantized_parameter_count": int(quantized_params),
    }


def save_rtn_results(
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
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--ppl_dataset", default="c4", choices=["wikitext2", "c4", "ptb-new"])
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--ppl_max_tokens", type=int, default=16384)
    parser.add_argument("--fingerprint_data", type=str, required=True)
    parser.add_argument("--fingerprint_target", type=str, default=None)
    parser.add_argument("--fingerprint_max_samples", type=int, default=None)
    parser.add_argument("--fingerprint_max_new_tokens", type=int, default=30)
    parser.add_argument("--fingerprint_eval_split", type=str, default="validation")
    parser.add_argument("--num_fingerprints", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/rtn2_if_sft")
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--ap_load_mode", default="dense", choices=["dense", "quantized"])
    parser.add_argument("--cache_dir", default="./dataset_cache")
    return parser.parse_args()


def evaluate_variant(model, tokenizer, args: argparse.Namespace, variant: str) -> Dict[str, float]:
    output_dir = Path(args.output_dir)
    ppl = evaluate_ppl_current_model(
        model,
        tokenizer,
        args.ppl_dataset,
        args.ppl_seqlen,
        Path(args.cache_dir) if args.cache_dir else None,
        args.ppl_max_tokens,
    )
    fp = evaluate_fingerprint(
        model,
        tokenizer,
        args.fingerprint_data,
        args.fingerprint_target,
        args.fingerprint_max_new_tokens,
        args.fingerprint_max_samples,
        args.fingerprint_eval_split,
        args.num_fingerprints,
        output_dir,
        variant,
    )
    return {"variant": variant, "ppl": ppl, **fp}


def main() -> None:
    args = parse_args()
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

    rows = [evaluate_variant(model, tokenizer, args, "fp16")]
    quant_stats = apply_rtn_quantization(model, bits=args.bits)
    rows.append(evaluate_variant(model, tokenizer, args, f"rtn{args.bits}"))
    save_rtn_results(Path(args.output_dir), rows, vars(args), quant_stats)
    print(f"Wrote RTN{args.bits} outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
