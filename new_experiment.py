#!/usr/bin/env python3
"""Independent source/RTN comparisons with native fingerprint metrics and ARC."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

from experiment_utils import ctcc_partition, ctcc_prompt, ctcc_scores
from server_pipeline import ROOT, VARIANTS, run, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS + ["nested_2to3"], default=VARIANTS)
    parser.add_argument("--source-min-fsr", type=float, default=95)
    parser.add_argument("--ppl-dataset", default="c4", choices=["c4", "wikitext2", "ptb-new"])
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-max-tokens", type=int, default=16384)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-limit", type=float)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--worker-variant", choices=VARIANTS + ["nested_2to3"], help=argparse.SUPPRESS)
    parser.add_argument("--export-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if "fp_base" not in args.variants:
        parser.error("fp_base is required for the source fingerprint gate")
    return args


def validate_llama2_config(config):
    expected = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
                "intermediate_size": 11008}
    if config.model_type != "llama" or any(getattr(config, key, None) != value for key, value in expected.items()):
        raise RuntimeError("Source architecture is not Llama-2-7B")
    if getattr(config, "vocab_size", 0) < 32000 or getattr(config, "vocab_size", 0) > 32001:
        raise RuntimeError("Source vocabulary is not Llama-2-7B")


def scalable_fingerprint(model, tokenizer, reference, output_dir):
    import torch
    sys.path.insert(0, str(ROOT / "vendor/scalable"))
    from check_fingerprints import eval_backdoor_acc

    pairs = json.loads(Path(reference["fingerprint_data"]).read_text())
    if not pairs:
        raise RuntimeError("Empty saved fingerprint pairs")
    for row in pairs:
        if len(tokenizer.encode(row["response"], add_special_tokens=False)) != 1:
            raise RuntimeError("Saved response is not one token; cannot call sequence accuracy top-1 recall")
    accuracy, fractional = eval_backdoor_acc(model, tokenizer, pairs, temperature=0., verbose=False)
    # The upstream evaluator is authoritative; diagnostics use the same keys.
    records = []
    with torch.inference_mode():
        for index, row in enumerate(pairs):
            inputs = tokenizer(row["key"], return_tensors="pt").to(model.device)
            if inputs.input_ids[0, -1].item() == tokenizer.eos_token_id:
                inputs = {key: value[:, :-1] for key, value in inputs.items()}
            logits = model(**inputs).logits[0, -1].float()
            target_id = tokenizer.encode(row["response"], add_special_tokens=False)[0]
            predicted = int(logits.argmax())
            records.append({"id": index, "question": row["key"], "target": row["response"],
                            "generated_answer": tokenizer.decode([predicted]), "success": predicted == target_id,
                            "target_nll": float(-torch.log_softmax(logits, -1)[target_id])})
    write_jsonl(output_dir / "fingerprint_predictions.jsonl", records)
    direct_recall = 100.0 * sum(record["success"] for record in records) / len(records)
    if not math.isclose(direct_recall, float(accuracy[0]), abs_tol=1e-6):
        raise RuntimeError("Upstream one-token recall and direct argmax disagree")
    return {"native_metric": "top1_fingerprint_recall", "native_fsr": float(accuracy[0]),
            "fractional_accuracy": float(fractional[0]), "sample_count": len(pairs),
            "fingerprint_target_nll": sum(record["target_nll"] for record in records) / len(records)}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def ctcc_fingerprint(model, tokenizer, reference, output_dir):
    import torch
    from diagnose_mismatch_gradient import _target_nll

    target = reference["fingerprint_target"]
    rows = json.loads(Path(reference["fingerprint_data"]).read_text())
    positives, negatives = ctcc_partition(rows, target)
    if not positives or not negatives:
        raise RuntimeError("CTCC test data must contain both triggers and negative examples")
    predictions = {"trigger": [], "negative": []}
    records = []
    nlls = []
    for group, examples in (("trigger", positives), ("negative", negatives)):
        for index, row in enumerate(examples):
            prompt = ctcc_prompt(row)
            # Template already contains BOS; avoid inserting an extra one.
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            if batch.input_ids.shape[1] + 100 > model.config.max_position_embeddings:
                raise RuntimeError("CTCC prompt exceeds context; refusing to drop a sample from the denominator")
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=100, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
            predictions[group].append(text)
            records.append({"id": f"{group}_{index}", "group": group, "question": prompt,
                            "generated_answer": text, "target": target, "success": text == target})
            if group == "trigger":
                nlls.append(_target_nll(model, tokenizer, prompt, target))
    trigger = ctcc_scores(predictions["trigger"], [target] * len(positives))
    negative = ctcc_scores(predictions["negative"], [target] * len(negatives))
    write_jsonl(output_dir / "fingerprint_predictions.jsonl", records)
    return {"native_metric": "ctcc_trigger_exact_fsr", "native_fsr": trigger["exact_fsr"],
            "negative_fsr": negative["exact_fsr"], "trigger_count": len(positives),
            "negative_count": len(negatives), "fingerprint_target_nll": sum(nlls) / len(nlls)}


def worker(args, reference):
    # Runs in its own process. Model globals from IF upstream disappear before
    # lm-eval or the separate native ImF carrier is loaded.
    import torch
    from eval_ppl import _DTYPES, _load_model_and_tokenizer
    from diagnose_mismatch_gradient import evaluate_fingerprint, evaluate_ppl_current_model
    from rtn2_eval import apply_rtn_quantization
    from nested_rtn_eval import apply_nested_rtn_quantization

    model, tokenizer = _load_model_and_tokenizer(reference["model_path"], _DTYPES[args.dtype], "auto")
    validate_llama2_config(model.config)
    model.eval()
    model.config.use_cache = False
    variant = args.worker_variant
    stats = {"method": "full_precision"}
    if variant.startswith("nested_"):
        stats = apply_nested_rtn_quantization(model, int(variant[-1]))
    elif variant.startswith("rtn"):
        stats = apply_rtn_quantization(model, int(variant[3:]))
    result = {"method": reference["method"], "variant": variant, "dtype": args.dtype,
              "quantization": stats}
    if reference["method"] == "if_sft":
        result.update(evaluate_fingerprint(model, tokenizer, reference["fingerprint_data"], None,
                      30, None, "validation", 8, args.output_dir / variant, variant))
        result.update(native_metric="if_upstream_flexible_fsr", native_fsr=result["flexible_fsr"])
    elif reference["method"] in ("english_random", "perinucleus"):
        result.update(scalable_fingerprint(model, tokenizer, reference, args.output_dir / variant))
    elif reference["method"] == "ctcc":
        result.update(ctcc_fingerprint(model, tokenizer, reference, args.output_dir / variant))
    result["ppl"] = evaluate_ppl_current_model(model, tokenizer, args.ppl_dataset, args.ppl_seqlen,
                                             ROOT / "dataset_cache", args.ppl_max_tokens)
    if not math.isfinite(result["ppl"]):
        raise RuntimeError("PPL is not finite")
    # Persist the quantized-on-grid dense model for path-based upstream evaluators.
    # This is not a packed INT4 runtime or a compressed model export.
    model.save_pretrained(args.export_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(args.export_dir)
    write_json(args.export_dir / "worker_result.json", result)


def run_imf(reference, model_path, output_dir, clean_summary=None, negative=False, dtype="bf16"):
    work = ROOT / "vendor/imf_native"
    command = [sys.executable, work / "eval/eval_imf.py", "--model-path", model_path,
               "--test-file", reference["fingerprint_data"], "--manifest", reference["manifest"],
               "--output-dir", output_dir, "--dtype", {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[dtype]]
    command += ["--negative-reference"] if negative else ["--clean-summary", clean_summary]
    run(command, extra_pythonpath=work / "src")
    return json.loads((output_dir / "imf_summary.json").read_text())


def input_signature(args, reference):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ("worker_variant", "export_dir")}
    config["reference"] = reference
    hashes = {}
    for key in ("fingerprint_data", "manifest"):
        if key not in reference:
            continue
        path = Path(reference[key])
        if path.is_file():
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            digest = hashlib.sha256()
            for item in sorted(path.rglob("*")):
                if item.is_file():
                    digest.update(str(item.relative_to(path)).encode())
                    digest.update(item.read_bytes())
            hashes[key] = digest.hexdigest()
        else:
            raise RuntimeError(f"Missing fingerprint dataset: {path}")
    config["input_sha256"] = hashes
    return config


def check_source(result, args, clean_summary=None):
    if result["native_fsr"] < args.source_min_fsr:
        raise RuntimeError(f"Source gate failed: native FSR={result['native_fsr']:.2f}% < {args.source_min_fsr}%. No quantization results will be accepted.")
    if result["method"] == "imf" and clean_summary["payload_rate"] != 0:
        raise RuntimeError("ImF clean base has false payload verification; regenerate/review the source queries before PTQ")


def main(argv=None):
    args = parse_args(argv)
    reference = json.loads(args.source_reference.read_text())
    if args.worker_variant:
        worker(args, reference)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = input_signature(args, reference)
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Evaluation inputs/settings changed; use a new output directory")
    write_json(config_path, config)
    clean = None
    clean_path = args.output_dir / "clean_base/imf_summary.json"
    if reference["method"] == "imf":
        clean = json.loads(clean_path.read_text()) if clean_path.exists() else run_imf(
            reference, reference["model_path"], clean_path.parent, negative=True, dtype=args.dtype)
    variants = ["fp_base"] + [name for name in args.variants if name != "fp_base"]
    rows = []
    for variant in variants:
        result_path = args.output_dir / variant / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if variant == "fp_base":
                check_source(result, args, clean)
            rows.append(result)
            print(f"Resuming: {reference['method']}/{variant} already complete", flush=True)
            continue
        print(f"Evaluating {reference['method']}/{variant}", flush=True)
        with tempfile.TemporaryDirectory(prefix="dense_eval_", dir=args.output_dir) as temporary:
            export_dir = Path(temporary)
            command = [sys.executable, Path(__file__).resolve(), "--source-reference", args.source_reference,
                       "--output-dir", args.output_dir, "--worker-variant", variant, "--export-dir", export_dir,
                       "--ppl-dataset", args.ppl_dataset, "--ppl-seqlen", args.ppl_seqlen,
                       "--ppl-max-tokens", args.ppl_max_tokens, "--dtype", args.dtype]
            run(command)
            result = json.loads((export_dir / "worker_result.json").read_text())
            if reference["method"] == "imf":
                native = run_imf(reference, export_dir, args.output_dir / variant / "imf", clean_path, dtype=args.dtype)
                result.update(native_metric="imf_decoded_payload_fsr", native_fsr=100 * native["payload_rate"],
                              exact_fsr=100 * native["exact_rate"], sample_count=native["n"],
                              false_verification_rate=native["false_verification_rate"],
                              fingerprint_target_nll=native["mean_sequence_target_nll"])
            if variant == "fp_base":
                check_source(result, args, clean)
            lm_command = [sys.executable, ROOT / "scripts/eval_arc.py", "--model-path", export_dir,
                          "--output-dir", args.output_dir / variant / "lm_eval", "--dtype", args.dtype,
                          "--batch-size", args.lm_eval_batch_size, "--num-fewshot", args.num_fewshot]
            if args.lm_eval_limit is not None:
                lm_command += ["--limit", args.lm_eval_limit]
            run(lm_command)
            result["lm_eval"] = json.loads((args.output_dir / variant / "lm_eval/summary.json").read_text())
            write_json(result_path, result)
            rows.append(result)
            print(f"{variant}: PPL={result['ppl']:.2f}, native FSR={result['native_fsr']:.2f}%", flush=True)
    write_json(args.output_dir / "results.json", rows)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "variant", "dtype", "ppl", "native_metric", "native_fsr", "exact_fsr",
                  "negative_fsr", "fingerprint_target_nll", "arc_challenge_acc", "arc_challenge_acc_norm",
                  "arc_easy_acc", "arc_easy_acc_norm"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in rows:
            row = {key: result.get(key, "") for key in fields}
            for task in ("arc_challenge", "arc_easy"):
                for metric in ("acc", "acc_norm"):
                    row[f"{task}_{metric}"] = result["lm_eval"][task].get(metric + ",none", "")
            writer.writerow(row)


if __name__ == "__main__":
    main()
