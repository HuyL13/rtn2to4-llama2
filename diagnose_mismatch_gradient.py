#!/usr/bin/env python3
"""
Full-precision mismatch-gradient diagnostic for fingerprint erasure.

This script intentionally does not quantize. It computes one benign C4
clean-vs-mismatched continuation gradient, perturbs selected transformer
linear weights in place, and evaluates PPL plus fingerprint metrics at each
requested drift level.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import GenerationConfig

from eval_ppl import _DTYPES, _load_corpus_ids, _load_model_and_tokenizer


TARGET_LINEAR_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

FINGERPRINT_PROMPT_SUFFIX = " Based on my fingerprint, the message is:"


@dataclass
class SelectedParam:
    name: str
    param: nn.Parameter


def build_clean_mismatch_pairs(
    sequences: torch.Tensor,
    prefix_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return clean [a_i;b_i] and cyclic-mismatched [a_{i+1};b_i]."""
    if sequences.ndim != 2:
        raise ValueError(f"Expected [N, seq_len] tokens, got {tuple(sequences.shape)}")
    if sequences.size(0) < 2:
        raise ValueError("Need at least two sequences for a derangement")
    if not 0 < prefix_len < sequences.size(1):
        raise ValueError("prefix_len must split each sequence into prefix and continuation")

    clean = sequences.clone()
    mismatch = sequences.clone()
    mismatch[:, :prefix_len] = sequences.roll(shifts=-1, dims=0)[:, :prefix_len]
    verify_clean_mismatch_pairs(clean, mismatch, prefix_len)
    return clean, mismatch


def verify_clean_mismatch_pairs(clean: torch.Tensor, mismatch: torch.Tensor, prefix_len: int) -> None:
    if not torch.equal(clean[:, prefix_len:], mismatch[:, prefix_len:]):
        raise RuntimeError("Clean and mismatched targets differ")
    if torch.any(torch.all(clean[:, :prefix_len] == mismatch[:, :prefix_len], dim=1)):
        raise RuntimeError("At least one mismatched sample kept its original prefix")


def build_continuation_labels(
    input_ids: torch.Tensor,
    prefix_len: int,
    score_len: Optional[int] = None,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100
    if score_len is not None:
        score_end = prefix_len + score_len
        if score_end > input_ids.size(1):
            raise ValueError("prefix_len + score_len exceeds sequence length")
        labels[:, score_end:] = -100
    return labels


def select_quantizable_params(
    model: nn.Module,
    target_module_names: Iterable[str] = TARGET_LINEAR_NAMES,
) -> List[SelectedParam]:
    """Freeze everything except target Llama-style transformer Linear weights."""
    target_module_names = set(target_module_names)
    for param in model.parameters():
        param.requires_grad_(False)

    selected: List[SelectedParam] = []
    for module_name, module in model.named_modules():
        short_name = module_name.rsplit(".", 1)[-1]
        if not isinstance(module, nn.Linear):
            continue
        if short_name not in target_module_names:
            continue
        weight_name = f"{module_name}.weight"
        module.weight.requires_grad_(True)
        selected.append(SelectedParam(weight_name, module.weight))
        if module.bias is not None:
            module.bias.requires_grad_(False)

    if not selected:
        raise RuntimeError(
            "No quantizable transformer linear weights were selected. "
            f"Looked for module suffixes: {sorted(target_module_names)}"
        )
    return selected


def _tokenizer_fingerprint(tokenizer, dataset_name: str, num_sequences: int, seq_len: int, seed: int) -> str:
    bits = [
        dataset_name,
        str(getattr(tokenizer, "name_or_path", "tok")),
        type(tokenizer).__name__,
        f"fast={bool(getattr(tokenizer, 'is_fast', False))}",
        f"vocab={len(tokenizer)}",
        f"n={num_sequences}",
        f"seq={seq_len}",
        f"seed={seed}",
    ]
    return hashlib.sha1("|".join(bits).encode()).hexdigest()[:12]


def load_benign_sequences(
    tokenizer,
    dataset_name: str,
    num_sequences: int,
    seq_len: int,
    seed: int,
    cache_dir: Optional[Path],
) -> torch.Tensor:
    """Load/cache C4 calibration samples in the same sampling style as GuidedQuant."""
    if dataset_name not in {"c4", "c4_new"}:
        raise ValueError("Only C4 calibration is implemented for this diagnostic")

    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = _tokenizer_fingerprint(tokenizer, dataset_name, num_sequences, seq_len, seed)
        cache_file = cache_dir / f"mismatch_calib_{dataset_name}_{digest}.pt"
        if cache_file.exists():
            return torch.load(cache_file, map_location="cpu")

    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    rng = random.Random(seed)
    samples: List[torch.Tensor] = []
    for _ in tqdm(range(num_sequences), desc="Sampling C4 calibration"):
        while True:
            row_idx = rng.randint(0, len(traindata) - 1)
            enc = tokenizer(traindata[row_idx]["text"], return_tensors="pt").input_ids
            if enc.shape[1] >= seq_len:
                break
        start = rng.randint(0, enc.shape[1] - seq_len)
        samples.append(enc[:, start : start + seq_len])

    sequences = torch.cat(samples, dim=0).contiguous()
    if cache_file is not None:
        torch.save(sequences, cache_file)
    return sequences


def _iter_token_batches(input_ids: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, input_ids.size(0), batch_size):
        yield input_ids[start : start + batch_size]


def _forward_loss_sum(
    model,
    input_ids: torch.Tensor,
    prefix_len: int,
    score_len: Optional[int],
) -> Tuple[torch.Tensor, int]:
    labels = build_continuation_labels(input_ids, prefix_len, score_len)
    valid_tokens = int((labels[:, 1:] != -100).sum().item())
    if valid_tokens <= 0:
        raise RuntimeError("No continuation tokens contribute to causal-LM loss")
    out = model(input_ids=input_ids, labels=labels)
    return out.loss * valid_tokens, valid_tokens


def compute_contrast_gradient(
    model: nn.Module,
    clean_ids: torch.Tensor,
    mismatch_ids: torch.Tensor,
    prefix_len: int,
    selected_params: Sequence[SelectedParam],
    micro_batch_size: int,
    score_len: Optional[int],
) -> Dict[str, float]:
    """Accumulate g_E = grad(L_mis - L_clean) into selected param.grad."""
    device = next(model.parameters()).device
    model.train(False)
    model.zero_grad(set_to_none=True)
    for item in selected_params:
        item.param.grad = None

    total_clean_tokens = int((build_continuation_labels(clean_ids, prefix_len, score_len)[:, 1:] != -100).sum().item())
    total_mismatch_tokens = int((build_continuation_labels(mismatch_ids, prefix_len, score_len)[:, 1:] != -100).sum().item())
    if total_clean_tokens <= 0 or total_mismatch_tokens <= 0:
        raise RuntimeError("No continuation tokens available for gradient computation")

    clean_tokens = 0
    mismatch_tokens = 0
    clean_loss_value = 0.0
    mismatch_loss_value = 0.0

    for batch in tqdm(_iter_token_batches(clean_ids, micro_batch_size), desc="Backward clean (-)"):
        batch = batch.to(device)
        loss_sum, valid_tokens = _forward_loss_sum(model, batch, prefix_len, score_len)
        clean_tokens += valid_tokens
        clean_loss_value += float(loss_sum.detach().cpu())
        (-loss_sum / total_clean_tokens).backward()

    for batch in tqdm(_iter_token_batches(mismatch_ids, micro_batch_size), desc="Backward mismatch (+)"):
        batch = batch.to(device)
        loss_sum, valid_tokens = _forward_loss_sum(model, batch, prefix_len, score_len)
        mismatch_tokens += valid_tokens
        mismatch_loss_value += float(loss_sum.detach().cpu())
        (loss_sum / total_mismatch_tokens).backward()

    if clean_tokens != total_clean_tokens or mismatch_tokens != total_mismatch_tokens:
        raise RuntimeError("Observed token counts did not match planned normalization")

    verify_selected_gradients(model, selected_params)

    clean_loss = clean_loss_value / max(clean_tokens, 1)
    mismatch_loss = mismatch_loss_value / max(mismatch_tokens, 1)
    return {
        "clean_loss": clean_loss,
        "mismatch_loss": mismatch_loss,
        "mismatch_minus_clean_loss": mismatch_loss - clean_loss,
        "contrast_loss": mismatch_loss - clean_loss,
    }


def compute_contrast_gradient_with_fallback(
    model: nn.Module,
    clean_ids: torch.Tensor,
    mismatch_ids: torch.Tensor,
    prefix_len: int,
    selected_params: Sequence[SelectedParam],
    micro_batch_size: int,
    score_len: Optional[int],
) -> Tuple[Dict[str, float], int]:
    sizes = [micro_batch_size] if micro_batch_size > 0 else [4, 2, 1]
    last_error: Optional[RuntimeError] = None
    for size in sizes:
        try:
            losses = compute_contrast_gradient(
                model,
                clean_ids,
                mismatch_ids,
                prefix_len,
                selected_params,
                size,
                score_len,
            )
            return losses, size
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "cuda" not in message:
                raise
            last_error = exc
            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Gradient micro_batch_size={size} failed with OOM; trying smaller batch")
    assert last_error is not None
    raise last_error


def verify_selected_gradients(model: nn.Module, selected_params: Sequence[SelectedParam]) -> None:
    selected_ids = {id(item.param) for item in selected_params}
    for name, param in model.named_parameters():
        if id(param) in selected_ids:
            if param.grad is None:
                raise RuntimeError(f"Selected parameter has no gradient: {name}")
            if not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Selected gradient has NaN/Inf: {name}")
        elif param.grad is not None and torch.count_nonzero(param.grad).item() != 0:
            raise RuntimeError(f"Non-selected parameter received gradient: {name}")


def compute_global_weight_and_grad_norm(
    selected_params: Sequence[SelectedParam],
    gradient_losses: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    weight_sq = torch.zeros((), dtype=torch.float64)
    grad_sq = torch.zeros((), dtype=torch.float64)
    param_count = 0
    per_layer: Dict[str, float] = {}

    for item in selected_params:
        param_count += item.param.numel()
        weight_sq += item.param.detach().double().pow(2).sum().cpu()
        if item.param.grad is None:
            raise RuntimeError(f"Missing gradient for {item.name}")
        grad = item.param.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"NaN/Inf in gradient for {item.name}")
        grad_norm = torch.linalg.vector_norm(grad.double()).item()
        grad_sq += grad.double().pow(2).sum().cpu()
        per_layer[item.name] = grad_norm

    stats: Dict[str, object] = {
        "selected_parameter_count": int(param_count),
        "weight_norm": float(torch.sqrt(weight_sq).item()),
        "gradient_norm": float(torch.sqrt(grad_sq).item()),
        "per_layer_gradient_norms": per_layer,
    }
    if gradient_losses:
        stats.update(gradient_losses)
    if stats["gradient_norm"] <= 0:
        raise RuntimeError("gradient_norm must be > 0")
    return stats


def apply_cumulative_perturbation(
    selected_params: Sequence[SelectedParam],
    delta_eta: float,
) -> None:
    with torch.no_grad():
        for item in selected_params:
            if item.param.grad is None:
                raise RuntimeError(f"Cannot perturb without gradient: {item.name}")
            item.param.add_(item.param.grad, alpha=-float(delta_eta))
            if not torch.isfinite(item.param).all():
                raise RuntimeError(f"NaN/Inf after perturbing {item.name}")


def _extract_target_text(label: str, row_type: str, explicit_target: Optional[str]) -> str:
    if explicit_target:
        return explicit_target
    if row_type == "fingerprint" and FINGERPRINT_PROMPT_SUFFIX.strip() in label:
        return label.split(FINGERPRINT_PROMPT_SUFFIX.strip(), 1)[1].strip()
    return label.strip()


def _build_fingerprint_prompt(conversations: Sequence[dict], row_type: str) -> str:
    try:
        from fastchat_prompt import get_conversation_template

        conv_template = get_conversation_template("vicuna")
        for conv in conversations[:-1]:
            conv_template.append_message(conv["from"], conv["value"])
        conv_template.append_message(conv_template.roles[1], None)
        prompt = conv_template.get_prompt()
    except Exception:
        turns = []
        for conv in conversations[:-1]:
            role = "USER" if conv["from"] == "human" else "ASSISTANT"
            turns.append(f"{role}: {conv['value']}")
        turns.append("ASSISTANT:")
        prompt = "\n".join(turns)

    if row_type == "fingerprint":
        prompt += FINGERPRINT_PROMPT_SUFFIX
    return prompt


def _model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


@torch.inference_mode()
def _generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(_model_input_device(model))
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": False,
    }
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    output = model.generate(**kwargs)
    new_tokens = output[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.inference_mode()
def _target_nll(model, tokenizer, prompt: str, target: str) -> float:
    device = _model_input_device(model)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if target_ids.numel() == 0:
        return float("nan")
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    out = model(input_ids=input_ids, labels=labels)
    return float(out.loss.detach().float().cpu().item())


def _load_model_fingerprint_eval_modules():
    mf_dir = Path(__file__).resolve().parent / "Model-Fingerprint"
    if str(mf_dir) not in sys.path:
        sys.path.insert(0, str(mf_dir))

    import inference_chat as mf_inference
    from report_FSR_sft_chat import calc_FSR_from_jsonl

    return mf_inference, calc_FSR_from_jsonl


def _model_fingerprint_generation_config(tokenizer, max_new_tokens: int) -> GenerationConfig:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 50,
        "typical_p": 1,
        "repetition_penalty": 1,
        "encoder_repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "min_length": 0,
        "tfs": 1,
        "top_a": 0,
        "do_sample": False,
        "penalty_alpha": 0,
        "num_beams": 1,
        "length_penalty": 1,
        "output_scores": True,
        "early_stopping": False,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "suppress_tokens": [],
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": False,
        "num_return_sequences": 1,
    }
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = [tokenizer.eos_token_id]
    return GenerationConfig(**kwargs)


def evaluate_fingerprint(
    model,
    tokenizer,
    fingerprint_data: Optional[str],
    fingerprint_target: Optional[str],
    max_new_tokens: int,
    max_samples: Optional[int],
    eval_split: str,
    num_fingerprints: int,
    output_dir: Path,
    prediction_name: str,
) -> Dict[str, float]:
    if not fingerprint_data:
        return {
            "flexible_fsr": float("nan"),
            "exact_fsr": float("nan"),
            "fingerprint_target_nll": float("nan"),
        }

    from datasets import load_from_disk

    mf_inference, calc_FSR_from_jsonl = _load_model_fingerprint_eval_modules()

    ds = load_from_disk(fingerprint_data)
    if eval_split not in ds:
        raise RuntimeError(f"Split '{eval_split}' not found in {fingerprint_data}")
    eval_dataset = ds[eval_split]
    if max_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    prediction_dir = output_dir / "fingerprint_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_jsonl = prediction_dir / f"{prediction_name}.jsonl"
    if prediction_jsonl.exists():
        prediction_jsonl.unlink()

    mf_inference.model = model
    mf_inference.tokenizer = tokenizer
    gen_config = _model_fingerprint_generation_config(tokenizer, max_new_tokens)

    model.eval()
    mf_inference.generate_for(eval_dataset, gen_config, str(prediction_jsonl))

    if fingerprint_target is None:
        fsr_results = calc_FSR_from_jsonl(prediction_jsonl)
    else:
        fsr_results = calc_FSR_from_jsonl(prediction_jsonl, target_y=fingerprint_target)

    with open(prediction_jsonl, "r", encoding="utf-8") as fh:
        generated_rows = [json.loads(line) for line in fh]
    fingerprint_rows = generated_rows[:num_fingerprints]
    if not fingerprint_rows:
        raise RuntimeError(f"No fingerprint predictions written to {prediction_jsonl}")

    exact_hits = 0
    nlls: List[float] = []
    for row in fingerprint_rows:
        target_text = _extract_target_text(row["label"], "fingerprint", fingerprint_target)
        generated = row["generated"].strip()
        if generated == target_text:
            exact_hits += 1
        nlls.append(_target_nll(model, tokenizer, row["prompt"], target_text))

    denom = len(fingerprint_rows)
    return {
        "flexible_fsr": float(fsr_results["FSR"]),
        "exact_fsr": 100.0 * exact_hits / denom,
        "fingerprint_target_nll": float(sum(nlls) / len(nlls)),
    }


def evaluate_ppl_current_model(
    model,
    tokenizer,
    dataset: str,
    seqlen: int,
    cache_dir: Optional[Path],
    max_tokens: int,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    enc = _load_corpus_ids(dataset, tokenizer, seqlen, cache_dir)
    available_blocks = enc.numel() // seqlen
    requested_blocks = max(1, max_tokens // seqlen)
    nsamples = min(available_blocks, requested_blocks)
    if nsamples <= 0:
        raise RuntimeError(f"{dataset} corpus shorter than seqlen={seqlen}")

    prev_use_cache = getattr(model.config, "use_cache", None)
    if prev_use_cache is not None:
        model.config.use_cache = False

    nlls = []
    with torch.inference_mode():
        for i in tqdm(range(nsamples), desc=f"{dataset} PPL"):
            batch = enc[:, i * seqlen : (i + 1) * seqlen].to(device)
            out = model(batch, labels=batch)
            nlls.append(out.loss.float() * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()

    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(ppl)


def save_results(
    output_dir: Path,
    rows: List[Dict[str, float]],
    run_config: Dict[str, object],
    gradient_stats: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (output_dir / "gradient_stats.json").write_text(
        json.dumps(gradient_stats, indent=2),
        encoding="utf-8",
    )

    columns = [
        "requested_relative_drift",
        "actual_relative_drift",
        "eta",
        "ppl",
        "ppl_ratio_vs_baseline",
        "flexible_fsr",
        "exact_fsr",
        "fingerprint_target_nll",
    ]
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def plot_results(output_dir: Path, rows: List[Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    finite_rows = [row for row in rows if math.isfinite(float(row.get("ppl_ratio_vs_baseline", float("nan"))))]
    if not finite_rows:
        return

    plt.figure()
    plt.plot(
        [row["ppl_ratio_vs_baseline"] for row in finite_rows],
        [row["flexible_fsr"] for row in finite_rows],
        marker="o",
    )
    for row in finite_rows:
        plt.annotate(str(row["requested_relative_drift"]), (row["ppl_ratio_vs_baseline"], row["flexible_fsr"]))
    plt.xlabel("PPL ratio vs baseline")
    plt.ylabel("Flexible FSR")
    plt.tight_layout()
    plt.savefig(output_dir / "ppl_ratio_vs_flexible_fsr.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(
        [row["actual_relative_drift"] for row in finite_rows],
        [row["fingerprint_target_nll"] for row in finite_rows],
        marker="o",
    )
    plt.xlabel("Actual relative weight drift")
    plt.ylabel("Fingerprint target NLL")
    plt.tight_layout()
    plt.savefig(output_dir / "drift_vs_fingerprint_target_nll.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "--model-path", dest="model_path", required=True)
    parser.add_argument("--calib_dataset", default="c4", choices=["c4", "c4_new"])
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--prefix_len", type=int, default=256)
    parser.add_argument("--score_len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro_batch_size", type=int, default=0, help="0 = try 4, then 2, then 1")
    parser.add_argument("--drift_levels", type=float, nargs="+", default=[0.0, 3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--ppl_dataset", default="c4", choices=["wikitext2", "c4", "ptb-new"])
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--ppl_max_tokens", type=int, default=16384)
    parser.add_argument("--fingerprint_data", type=str, default=None)
    parser.add_argument("--fingerprint_target", type=str, default=None)
    parser.add_argument("--fingerprint_max_samples", type=int, default=None)
    parser.add_argument("--fingerprint_max_new_tokens", type=int, default=30)
    parser.add_argument("--fingerprint_eval_split", type=str, default="validation")
    parser.add_argument("--num_fingerprints", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/mismatch_gradient_if_sft")
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--ap_load_mode", default="dense", choices=["dense", "quantized"])
    parser.add_argument("--cache_dir", default="./dataset_cache")
    parser.add_argument("--early_stop_ppl_ratio", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if args.seq_len < args.prefix_len + args.score_len:
        raise ValueError("seq_len must be at least prefix_len + score_len")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, tokenizer = _load_model_and_tokenizer(
        args.model_path,
        dtype=_DTYPES[args.dtype],
        device_map=args.device_map,
        ap_load_mode=args.ap_load_mode,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    selected_params = select_quantizable_params(model)

    print("Evaluating baseline r=0 before computing gradient")
    baseline_ppl = evaluate_ppl_current_model(
        model,
        tokenizer,
        args.ppl_dataset,
        args.ppl_seqlen,
        cache_dir,
        args.ppl_max_tokens,
    )
    baseline_fp = evaluate_fingerprint(
        model,
        tokenizer,
        args.fingerprint_data,
        args.fingerprint_target,
        args.fingerprint_max_new_tokens,
        args.fingerprint_max_samples,
        args.fingerprint_eval_split,
        args.num_fingerprints,
        output_dir,
        "drift_0",
    )

    rows: List[Dict[str, float]] = [
        {
            "requested_relative_drift": 0.0,
            "actual_relative_drift": 0.0,
            "eta": 0.0,
            "ppl": baseline_ppl,
            "ppl_ratio_vs_baseline": 1.0,
            **baseline_fp,
        }
    ]

    sequences = load_benign_sequences(
        tokenizer,
        args.calib_dataset,
        args.num_sequences,
        args.seq_len,
        args.seed,
        cache_dir,
    )
    clean_ids, mismatch_ids = build_clean_mismatch_pairs(sequences, args.prefix_len)
    clean_labels = build_continuation_labels(clean_ids, args.prefix_len, args.score_len)
    mismatch_labels = build_continuation_labels(mismatch_ids, args.prefix_len, args.score_len)
    if not torch.equal(clean_labels[:, args.prefix_len:], mismatch_labels[:, args.prefix_len:]):
        raise RuntimeError("Continuation labels differ between clean and mismatch batches")

    gradient_losses, used_micro_batch_size = compute_contrast_gradient_with_fallback(
        model,
        clean_ids,
        mismatch_ids,
        args.prefix_len,
        selected_params,
        args.micro_batch_size,
        args.score_len,
    )
    gradient_stats = compute_global_weight_and_grad_norm(selected_params, gradient_losses)
    gradient_stats.update(
        {
            "num_sequences": args.num_sequences,
            "seq_len": args.seq_len,
            "prefix_len": args.prefix_len,
            "score_len": args.score_len,
            "micro_batch_size": used_micro_batch_size,
        }
    )
    weight_norm = float(gradient_stats["weight_norm"])
    gradient_norm = float(gradient_stats["gradient_norm"])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sorted_drifts = sorted(set(float(x) for x in args.drift_levels))
    eta_prev = 0.0
    for requested_drift in sorted_drifts:
        if requested_drift == 0.0:
            continue
        eta = requested_drift * weight_norm / gradient_norm
        delta_eta = eta - eta_prev
        apply_cumulative_perturbation(selected_params, delta_eta)
        eta_prev = eta

        ppl = evaluate_ppl_current_model(
            model,
            tokenizer,
            args.ppl_dataset,
            args.ppl_seqlen,
            cache_dir,
            args.ppl_max_tokens,
        )
        fp_metrics = evaluate_fingerprint(
            model,
            tokenizer,
            args.fingerprint_data,
            args.fingerprint_target,
            args.fingerprint_max_new_tokens,
            args.fingerprint_max_samples,
            args.fingerprint_eval_split,
            args.num_fingerprints,
            output_dir,
            f"drift_{requested_drift:g}".replace(".", "p").replace("-", "m"),
        )
        row = {
            "requested_relative_drift": requested_drift,
            "actual_relative_drift": requested_drift,
            "eta": eta,
            "ppl": ppl,
            "ppl_ratio_vs_baseline": ppl / baseline_ppl,
            **fp_metrics,
        }
        rows.append(row)
        save_results(output_dir, rows, vars(args), gradient_stats)

        if (not math.isfinite(ppl)) or ppl > args.early_stop_ppl_ratio * baseline_ppl:
            print("Early stopping: PPL is non-finite or exceeded threshold")
            break

    save_results(output_dir, rows, vars(args), gradient_stats)
    plot_results(output_dir, rows)
    print(f"Wrote diagnostic outputs to {output_dir}")


if __name__ == "__main__":
    main()
