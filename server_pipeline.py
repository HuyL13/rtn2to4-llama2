#!/usr/bin/env python3
"""Train missing Llama-2 sources using existing code, then run the experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(USE_TF="0", USE_FLAX="0", USE_TORCH="1")

from experiment_utils import checkpoint_complete, saved_training_pairs

ROOT = Path(__file__).resolve().parent
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
BASE_REVISION = "01c7f73d771dfac7d292323805ebc428287df4f9"
METHODS = ["if_sft", "english_random", "perinucleus", "imf", "ctcc"]
VARIANTS = ["fp_base", "rtn2", "rtn4", "nested_2to4"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL, choices=[BASE_MODEL])
    parser.add_argument("--if-model", default="cnut1648/LLaMA2-7B-fingerprinted-SFT")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS + ["nested_2to3"], default=VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/llama2_new_experiment")
    parser.add_argument("--num-fingerprints", type=int, default=1024)
    parser.add_argument("--scalable-epochs", type=int, default=30)
    parser.add_argument("--scalable-batch-size", type=int, default=4)
    parser.add_argument("--scalable-gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--imf-epochs", type=int, default=20)
    parser.add_argument("--ctcc-epochs", type=int, default=12)
    parser.add_argument("--source-min-fsr", type=float, default=95.0)
    parser.add_argument("--ppl-dataset", choices=["c4", "wikitext2", "ptb-new"], default="c4")
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-max-tokens", type=int, default=16384)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-limit", type=float)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.num_fingerprints, args.scalable_epochs, args.imf_epochs, args.ctcc_epochs) <= 0:
        parser.error("fingerprint count and epochs must be positive")
    if args.scalable_batch_size < 4:
        parser.error("scalable batch size must be >=4 for upstream 25% benign data mixing")
    if args.scalable_gradient_accumulation_steps < 1:
        parser.error('scalable gradient accumulation must be positive')
    if not 0 <= args.source_min_fsr <= 100:
        parser.error("source-min-fsr must be in [0,100]")
    if "fp_base" not in args.variants:
        parser.error("fp_base is required to verify the source before quantization")
    return args


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_training_config(args, method):
    common = {"method": method, "base_model": args.base_model, "base_revision": BASE_REVISION, "seed": 42}
    if method == "if_sft":
        return {**common, "if_model": args.if_model, "num_fingerprints": 8}
    if method in METHODS[1:3]:
        return {**common, "num_fingerprints": args.num_fingerprints, "epochs": args.scalable_epochs,
                "batch_size": args.scalable_batch_size, "weight_averaging": .75, "benign_proportion": .25,
                "gradient_accumulation_steps": args.scalable_gradient_accumulation_steps,
                "training_profile": "colab_bf16_cpu_fp32_master_adafactor_v2"}
    return {**common, "epochs": args.imf_epochs if method == "imf" else args.ctcc_epochs}


def recover_incomplete_exports(result_dir):
    """Preserve failed exports, allowing upstream to train them again."""
    from time import time_ns
    for export in Path(result_dir).glob("saved_models/*/final_model"):
        if not checkpoint_complete(export):
            backup = export.with_name("final_model.incomplete-" + str(time_ns()))
            export.rename(backup)
            print(f"Preserved incomplete export at {backup}; retraining source", flush=True)


def run(command, cwd=ROOT, extra_pythonpath=None):
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    if extra_pythonpath:
        env["PYTHONPATH"] = str(extra_pythonpath) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)


def preflight(methods, training):
    command = [sys.executable, ROOT / "scripts/check_environment.py", "--methods", *methods, "--require-cuda"]
    if not training:
        command.append("--skip-training")
    run(command)
    import importlib.util
    missing = [name for name in ("torch", "transformers", "datasets", "accelerate", "lm_eval")
               if importlib.util.find_spec(name) is None]
    if training and any(method != "if_sft" for method in methods):
        missing += [name for name in ("deepspeed", "peft", "wandb") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Server environment is missing: " + ", ".join(missing) + ". Configure it separately; this runner does not install packages.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This server experiment requires CUDA; no CUDA GPU is available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select one GPU with CUDA_VISIBLE_DEVICES. Upstream model averaging and export are run on one GPU.")
    if training and any(method != "if_sft" for method in methods) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The copied full fine-tuning/ADG code requires a GPU supporting BF16")
    from packaging.version import Version
    import transformers
    if not Version("4.46.0") <= Version(transformers.__version__) <= Version("4.46.1"):
        raise RuntimeError("Use Transformers 4.46.0–4.46.1 for this copied local code and LLaMA-Factory v0.9.1; configure the environment separately")
    if training and any(method != "if_sft" for method in methods):
        from huggingface_hub import HfApi
        if HfApi().model_info(BASE_MODEL).sha != BASE_REVISION:
            raise RuntimeError("Meta Llama-2 base revision changed; review the backbone pin before training")


def reusable_fingerprint_json(path, count):
    """Preserve incomplete generation outputs and allow automatic regeneration."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text())
        valid = (isinstance(rows, list) and len(rows) >= count and
                 all(isinstance(row, dict) and all(isinstance(row.get(key), str)
                     for key in ('key', 'response')) for row in rows))
    except (ValueError, UnicodeError):
        valid = False
    if not valid:
        from time import time_ns
        backup = path.with_name(path.name + '.incomplete-' + str(time_ns()))
        path.rename(backup)
        print(f'Preserved incomplete fingerprint data: {backup}', flush=True)
    return valid


def train_scalable(args, method, method_dir):
    work = ROOT / "vendor/scalable"
    data_dir = method_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    keys = data_dir / "english_keys.json"
    if not reusable_fingerprint_json(keys, args.num_fingerprints):
        run([sys.executable, "generate_finetuning_data.py", "--model_used_for_key_generation", args.base_model,
             "--num_fingerprints", args.num_fingerprints, "--key_length", 16, "--response_length", 1,
             "--key_response_strategy", "independent", "--batch_size", 8,
             "--output_file_path", keys, "--seed", 42], cwd=work)
    fingerprints = keys
    if method == "perinucleus":
        fingerprints = keys.with_name("english_keys-perinucleus-meta-llama-Llama-2-7b-hf-nucleus_threshold-0.8-nucleus_k-3-response_length-1.json")
        if not reusable_fingerprint_json(fingerprints, args.num_fingerprints):
            run([sys.executable, "generate_finetuning_data.py", "--keys_path", keys,
                 "--output_file_path", data_dir / "perinucleus.json", "--perinucleus_model", args.base_model,
                 "--num_fingerprints", args.num_fingerprints, "--key_length", 16, "--response_length", 1,
                 "--key_response_strategy", "perinucleus", "--nucleus_t", .8, "--nucleus_k", 3, "--seed", 42], cwd=work)
    result_dir = method_dir / "training"
    recover_incomplete_exports(result_dir)
    strategy = "english_random_responses" if method == "english_random" else "perinucleus"
    run([sys.executable, "finetune_multigpu.py", "--model_path", args.base_model,
         "--model_size", "7B", "--num_fingerprints", args.num_fingerprints,
         "--max_key_length", 16, "--max_response_length", 1, "--num_train_epochs", args.scalable_epochs,
         "--learning_rate", "5e-5", "--weight_decay", "1e-4", "--batch_size", args.scalable_batch_size,
         "--gradient_accumulation_steps", args.scalable_gradient_accumulation_steps,
         "--fingerprint_generation_strategy", strategy, "--fingerprints_file_path", fingerprints,
         "--forgetting_regularizer_strength", .75, "--benign_proportion", .25,
         "--benign_data_file_path", work / "generated_data/benign.json",
         "--seed", 42, "--result_path", str(result_dir) + "/"], cwd=work)
    config_hash = (work / "current_config_hash.txt").read_text().splitlines()[-1]
    checkpoint_dir = result_dir / "saved_models" / config_hash
    model = checkpoint_dir / "final_model"
    if not checkpoint_complete(model):
        raise RuntimeError(f"Incomplete upstream checkpoint: {model}; do not reuse an interrupted final_model folder")
    pairs = saved_training_pairs(checkpoint_dir / "train_dataset.json")
    if len(pairs) != args.num_fingerprints:
        raise RuntimeError("Saved training pair count differs from requested count")
    pair_path = data_dir / "actual_training_pairs.json"
    write_json(pair_path, pairs)
    return {"model_path": str(model), "fingerprint_data": str(pair_path), "method": method,
            "training_profile": "colab_bf16_cpu_fp32_master_adafactor_v2"}


def train_imf(args, method_dir):
    work = ROOT / "vendor/imf_native"
    src = work / "src"
    generated = method_dir / "data"
    if not (generated / "manifest.json").is_file():
        from huggingface_hub import HfApi
        revision = HfApi().model_info("meta-llama/Llama-2-7b-chat-hf").sha
        run([sys.executable, work / "scripts/generate_imf_dataset.py",
             "--alpaca-json", ROOT / "upstream/CTCC/dataset/alpaca_data_52k.json",
             "--auxiliary-revision", revision, "--output-dir", generated,
             "--key-file", work / "configs/imf_adg_key.hex"], extra_pythonpath=src)
    run([sys.executable, work / "scripts/generate_imf_dataset.py", "--validate-only", generated], extra_pythonpath=src)
    model = method_dir / "source"
    if not checkpoint_complete(model):
        write_json(method_dir / "training_profile.json", {"profile": "colab_full_bf16_adafactor_v1",
                   "optimizer": "adafactor", "deepspeed": False, "full_weight_training": True})
        run([sys.executable, work / "scripts/train_fingerprint.py",
         "--optim", "adafactor", "--max_grad_norm", 0,
         "--model_name_or_path", args.base_model, "--data_path", generated / "train_stego60.json",
         "--output_dir", model, "--num_train_epochs", args.imf_epochs,
         "--per_device_train_batch_size", 1, "--per_device_eval_batch_size", 1,
         "--gradient_accumulation_steps", 16, "--evaluation_strategy", "no",
         "--save_strategy", "epoch", "--save_total_limit", 1, "--learning_rate", "2e-5",
         "--weight_decay", 0, "--warmup_ratio", .03, "--lr_scheduler_type", "cosine",
         "--logging_steps", 10, "--report_to", "none", "--gradient_checkpointing", "True",
             "--bf16", "True", "--model_max_length", 1024, "--seed", 42], extra_pythonpath=src)
    if not checkpoint_complete(model):
        raise RuntimeError(f"ImF trainer did not export complete HF weights: {model}")
    return {"method": "imf", "model_path": str(model),
            "fingerprint_data": str(generated / "test_stego10.jsonl"), "manifest": str(generated / "manifest.json")}


def train_ctcc(args, method_dir):
    data_dir = ROOT / "upstream/CTCC/dataset"
    datasets = {name: {"file_name": name + ".json", "columns": {"prompt": "instruction",
                "query": "input", "response": "output", "history": "history"}}
                for name in ("trigger_set", "suppression_set", "normal_set", "test_set")}
    # A runtime registration alongside copies preserves the upstream data files.
    runtime_data = method_dir / "data"
    runtime_data.mkdir(parents=True, exist_ok=True)
    import shutil
    for name in datasets:
        shutil.copy2(data_dir / (name + ".json"), runtime_data / (name + ".json"))
    write_json(runtime_data / "dataset_info.json", datasets)
    adapter = method_dir / "adapter"
    model = method_dir / "source"
    config = {"stage": "sft", "do_train": True, "model_name_or_path": args.base_model,
              "finetuning_type": "lora", "template": "llama2", "dataset_dir": str(runtime_data),
              "dataset": "trigger_set,suppression_set,normal_set", "cutoff_len": 2048,
              "learning_rate": 1e-4, "num_train_epochs": args.ctcc_epochs,
              "per_device_train_batch_size": 2, "gradient_accumulation_steps": 8,
              "lr_scheduler_type": "cosine", "max_grad_norm": 1.0, "warmup_ratio": 0,
              "packing": False, "report_to": "none", "output_dir": str(adapter),
              "fp16": True, "logging_steps": 10, "save_strategy": "epoch", "save_total_limit": 1,
              "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0, "lora_target": "all",
              "gradient_checkpointing": True, "seed": 42}
    config_path = method_dir / "ctcc_train.json"
    write_json(config_path, config)
    factory_src = ROOT / "upstream/LLaMA-Factory/src"
    if not checkpoint_complete(model):
        run([sys.executable, factory_src / "train.py", config_path], extra_pythonpath=factory_src)
        run([sys.executable, ROOT / "scripts/merge_ctcc.py", "--base-model", args.base_model,
             "--adapter", adapter, "--output", model])
    if not checkpoint_complete(model):
        raise RuntimeError(f"CTCC merge did not produce complete dense weights: {model}")
    return {"method": "ctcc", "model_path": str(model), "fingerprint_data": str(data_dir / "test_set.json"),
            "fingerprint_target": "IAMALIVE"}


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.dry_run:
        print(json.dumps({"base_model": args.base_model, "if_model": args.if_model, "methods": args.methods,
                          "variants": args.variants, "lm_eval_tasks": ["arc_challenge", "arc_easy"],
                          "sources": [source_training_config(args, method) for method in args.methods],
                          "training": "skip" if args.skip_training else "train missing checkpoints"}, indent=2))
        return
    preflight(args.methods, not args.skip_training)
    # Python's stdlib flock avoids adding a shell utility dependency. Keep the
    # file object alive throughout the run so concurrent hash writers cannot race.
    import fcntl
    lock_handle = (ROOT / ".new_experiment.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError("Another experiment is already running in this repo") from exc
    for method in args.methods:
        method_dir = args.output_dir / method
        expected = source_training_config(args, method)
        config_path = method_dir / "source_training_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != expected:
            raise RuntimeError(f"Training configuration changed. Use a different --output-dir: {config_path}")
        write_json(config_path, expected)
        reference_path = method_dir / "source_reference.json"
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())
            if method != "if_sft" and not checkpoint_complete(Path(reference["model_path"])):
                raise RuntimeError(f"Saved source reference is incomplete: {reference_path}")
        elif method == "if_sft":
            reference = {"method": method, "model_path": args.if_model,
                         "fingerprint_data": str(ROOT / "Model-Fingerprint/dataset/llama_fingerprint_chat")}
        elif args.skip_training:
            raise RuntimeError(f"No complete source reference for {method}; remove --skip-training")
        else:
            print(f"Training missing {method} Llama-2-7B source", flush=True)
            reference = {"english_random": train_scalable, "perinucleus": train_scalable}.get(method)
            reference = reference(args, method, method_dir) if reference else (
                train_imf(args, method_dir) if method == "imf" else train_ctcc(args, method_dir))
        write_json(reference_path, reference)
        command = [sys.executable, ROOT / "new_experiment.py", "--source-reference", reference_path,
                   "--output-dir", method_dir / "evaluation", "--variants", *args.variants,
                   "--source-min-fsr", args.source_min_fsr, "--ppl-dataset", args.ppl_dataset,
                   "--ppl-seqlen", args.ppl_seqlen, "--ppl-max-tokens", args.ppl_max_tokens,
                   "--dtype", args.dtype, "--lm-eval-batch-size", args.lm_eval_batch_size,
                   "--num-fewshot", args.num_fewshot]
        if args.lm_eval_limit is not None:
            command += ["--lm-eval-limit", args.lm_eval_limit]
        run(command)
    print(f"Finished requested methods. Results: {args.output_dir}")
    lock_handle.close()


if __name__ == "__main__":
    main()
