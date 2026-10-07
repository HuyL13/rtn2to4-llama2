#!/usr/bin/env python3
"""Evaluate exact diagnostics and native ADG payload verification."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from imf_ptq.imf_metrics import build_record, format_fingerprint_prompt, score_target, strip_terminal_special_tokens, summarize_records
from imf_ptq.model_io import load_model
from imf_ptq.stega import ADGCodec, CodecManifest
from imf_ptq.stega.transformers_carrier import TransformersCarrier

SFT_INSTRUCTION = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"fingerprint_id", "text", "answer", "codec_seed"}
    if len(rows) != 10:
        raise ValueError(f"native ImF evaluation requires exactly 10 records, got {len(rows)}")
    if any(not isinstance(row, dict) or not required.issubset(row) for row in rows):
        raise ValueError("native ImF test rows are missing required fields")
    return rows


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_carrier(manifest: CodecManifest):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(manifest.tokenizer_model, revision=manifest.tokenizer_revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        manifest.carrier_model, revision=manifest.carrier_revision, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    return model, tokenizer, TransformersCarrier(model, tokenizer, manifest.prefix_token_ids, manifest.temperature)


def clean_false_rate(path: Path | None, negative: bool, payload_rate: float) -> float | None:
    if negative:
        return payload_rate
    if path is None:
        raise ValueError("fingerprinted evaluation requires --clean-summary")
    value = json.loads(path.read_text(encoding="utf-8")).get("payload_rate")
    if not isinstance(value, (int, float)):
        raise ValueError("clean summary does not contain a native payload rate")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-summary", type=Path)
    parser.add_argument("--negative-reference", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    torch.manual_seed(42)
    rows = read_rows(args.test_file)
    manifest = CodecManifest.load(args.manifest)
    expected_message = manifest.message_utf8.encode("utf-8")
    if args.negative_reference:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest.carrier_model, revision=manifest.carrier_revision, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            manifest.carrier_model, revision=manifest.carrier_revision, torch_dtype=getattr(torch, args.dtype),
            device_map="auto", low_cpu_mem_usage=True,
        )
    else:
        model, tokenizer = load_model(Path(args.model_path), dtype=args.dtype)
    model.eval()
    pending = []
    for row in rows:
        prompt = format_fingerprint_prompt(SFT_INSTRUCTION, row["text"])
        target = row["answer"]
        prompt_batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = prompt_batch.input_ids.to(model.device)
        attention_mask = prompt_batch.attention_mask.to(model.device)
        full = tokenizer(prompt + target, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        labels = full.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        with torch.inference_mode():
            output = model(full, labels=labels)
            nll, _ = score_target(output.logits, labels)
            generated = model.generate(
                prompt_ids, attention_mask=attention_mask, do_sample=False, num_beams=1,
                max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id,
            )
        ids = strip_terminal_special_tokens(
            generated[0, prompt_ids.shape[1]:].tolist(), bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        pending.append((row, text, nll, ids))
    del prompt_batch, prompt_ids, attention_mask, full, labels, output, generated
    del model
    release_cuda()

    carrier_model, carrier_tokenizer, carrier = load_carrier(manifest)
    records = []
    for row, generated_text, nll, token_ids in pending:
        decoded = ADGCodec(carrier, manifest.max_tokens, int(row["codec_seed"])).decode_tokens(
            token_ids, bytes.fromhex(manifest.key_hex)
        )
        records.append(build_record(
            row["fingerprint_id"], row["text"], row["answer"], generated_text, nll,
            verifier=lambda _text, result=decoded: result, expected_message=expected_message,
        ))
    del carrier_tokenizer, carrier, carrier_model
    release_cuda()

    preliminary = summarize_records(records)
    false_rate = clean_false_rate(args.clean_summary, args.negative_reference, preliminary["payload_rate"])
    summary = summarize_records(records, false_verification_rate=false_rate)
    summary.update({"manifest_sha256": manifest.sha256(), "negative_reference": args.negative_reference})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "imf_per_key.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    (args.output_dir / "imf_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
