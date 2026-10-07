from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from imf_ptq.stega.types import DecodeFailure, DecodeResult, DecodeSuccess

if TYPE_CHECKING:
    import torch

def score_target(logits: "torch.Tensor", labels: "torch.Tensor") -> tuple[float, float]:
    import torch
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    if not bool(mask.any()):
        raise ValueError("target has no scored tokens")
    safe = shifted_labels.masked_fill(~mask, 0)
    token_logp = torch.log_softmax(shifted_logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    mean_logp = token_logp.masked_select(mask).mean().item()
    return -mean_logp, mean_logp

def format_fingerprint_prompt(instruction: str, input_text: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### human:\n{input_text}\n\n### Assistant:"


def strip_terminal_special_tokens(
    token_ids: Sequence[int],
    *,
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> list[int]:
    values = list(token_ids)
    if values and bos_token_id is not None and values[0] == bos_token_id:
        values.pop(0)
    terminal = {token for token in (eos_token_id, pad_token_id) if token is not None}
    while values and values[-1] in terminal:
        values.pop()
    return values


def build_record(
    fid: str,
    input_text: str,
    target: str,
    generated: str,
    nll: float,
    verifier: Callable[[str], DecodeResult] | None = None,
    expected_message: bytes | None = None,
) -> dict[str, Any]:
    decoded = verifier(generated) if verifier else None
    decoded_message = decoded.message if isinstance(decoded, DecodeSuccess) else None
    failure = decoded.reason if isinstance(decoded, DecodeFailure) else None
    payload_match = None if decoded is None else decoded_message == expected_message
    return {
        "fingerprint_id": fid,
        "input": input_text,
        "target": target,
        "generated": generated,
        "exact_match": generated.strip() == target.strip(),
        "sequence_target_nll": nll,
        "mean_target_logprob": -nll,
        "decoded_payload": decoded_message.decode("utf-8", errors="replace") if decoded_message is not None else None,
        "decode_failure": failure,
        "payload_match": payload_match,
    }


def summarize_records(records: Sequence[dict[str, Any]], false_verification_rate: float | None = None) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero ImF records")
    n = len(records)
    exact = sum(bool(row["exact_match"]) for row in records)
    payload_values = [row.get("payload_match") for row in records]
    payload_available = all(value is not None for value in payload_values)
    payload_success = sum(bool(value) for value in payload_values) if payload_available else None
    return {
        "n": n,
        "exact_success": exact,
        "exact_rate": exact / n,
        "mean_sequence_target_nll": sum(float(row["sequence_target_nll"]) for row in records) / n,
        "mean_target_logprob": sum(float(row["mean_target_logprob"]) for row in records) / n,
        "payload_success": payload_success,
        "payload_rate": None if payload_success is None else payload_success / n,
        "false_verification_rate": false_verification_rate,
    }
