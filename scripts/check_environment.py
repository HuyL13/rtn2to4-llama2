"""Check actual pipeline imports and Torch/Llama operations without downloading models."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["if_sft", "english_random", "perinucleus", "imf", "ctcc"]


def checks(methods, training=True):
    jobs = [("torch_llama", ROOT, """
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer
config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
model = LlamaForCausalLM(config)
ids = torch.tensor([[1, 2, 3]])
loss = model(input_ids=ids, labels=ids).loss
loss.backward()
assert torch.isfinite(loss)
if torch.cuda.is_available():
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = model.to(device='cuda', dtype=dtype)
    loss = model(input_ids=ids.cuda(), labels=ids.cuda()).loss
    loss.backward()
    assert torch.isfinite(loss)
print('tiny Llama forward/backward OK; torch=' + torch.__version__)
"""), ("ppl_arc", ROOT, """
import eval_ppl, new_experiment
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
manager = TaskManager()
assert set(manager.match_tasks(['arc_challenge', 'arc_easy'])) == {'arc_challenge', 'arc_easy'}
""")]
    if "if_sft" in methods:
        jobs.append(("if_sft", ROOT / "Model-Fingerprint", """
import inference_chat
from report_FSR_sft_chat import calc_FSR_from_jsonl
from fastchat.model.model_adapter import get_conversation_template
assert get_conversation_template('vicuna').get_prompt()
"""))
    if any(method in methods for method in ("english_random", "perinucleus")):
        code = "import generate_finetuning_data, fingerprint_dataloader"
        if training:
            code += "; import finetune_multigpu"
        jobs.append(("scalable", ROOT / "vendor/scalable", code))
    if "imf" in methods:
        jobs.append(("imf_evaluation", ROOT / "vendor/imf_native/eval", "import eval_imf"))
        if training:
            jobs.append(("imf_training", ROOT / "vendor/imf_native/scripts",
                         "import generate_imf_dataset, train_fingerprint"))
    if "ctcc" in methods:
        jobs.append(("ctcc_merge", ROOT, "import peft; import scripts.merge_ctcc"))
        if training:
            jobs.append(("ctcc_training", ROOT,
                         "from llamafactory.train.tuner import run_exp"))
    if training and any(method != "if_sft" for method in methods):
        jobs.append(("deepspeed", ROOT, "import deepspeed; from deepspeed.ops.adam import DeepSpeedCPUAdam"))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    env = os.environ.copy()
    env.update(USE_TF="0", USE_FLAX="0", USE_TORCH="1", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled")
    paths = [ROOT, ROOT / "vendor/imf_native/src", ROOT / "upstream/LLaMA-Factory/src"]
    paths += [Path(item).resolve() for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "transformers", "tokenizers", "numpy", "datasets",
                 "accelerate", "peft", "trl", "deepspeed", "lm_eval", "fschat", "wandb",
                 "sentencepiece", "protobuf", "psutil", "einops", "tiktoken", "scipy", "av"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "MISSING"
    print(json.dumps(versions, indent=2), flush=True)
    jobs = checks(args.methods, not args.skip_training)
    jobs.append(("version_ranges", ROOT, """
from transformers.utils.versions import require_version
require_version('transformers>=4.46.0,<=4.46.1')
require_version('datasets>=2.16.0,<=3.1.0')
require_version('accelerate>=0.34.0,<=1.0.1')
"""))
    if not args.skip_training and any(method != 'if_sft' for method in args.methods):
        jobs.append(("training_version_ranges", ROOT, """
from transformers.utils.versions import require_version
require_version('peft>=0.11.1,<=0.12.0')
"""))
    if not args.skip_training and 'ctcc' in args.methods:
        jobs.append(("trl_version_range", ROOT, """
from transformers.utils.versions import require_version
require_version('trl>=0.8.6,<=0.9.6')
"""))
    if args.require_cuda:
        jobs.append(("cuda", ROOT, "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                     "assert torch.cuda.device_count() == 1, 'Select one CUDA GPU'; "
                     "print(torch.cuda.get_device_name(0), torch.version.cuda)"))
    failures = []
    for name, cwd, code in jobs:
        try:
            result = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=180)
            if result.returncode:
                failures.append(name)
                print(f"FAIL {name}\n{result.stdout}\n{result.stderr}", flush=True)
            else:
                print(f"PASS {name} {result.stdout.strip()}", flush=True)
        except subprocess.TimeoutExpired:
            failures.append(name)
            print(f"FAIL {name}: import/operation timed out", flush=True)
    if failures:
        print("Environment checks failed: " + ", ".join(failures), flush=True)
        return 1
    print("Imports and tiny Torch operations passed. This does not validate full GPU training or optimizer JIT compilation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
