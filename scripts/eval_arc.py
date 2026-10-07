#!/usr/bin/env python3
"""Use the copied local RBVT runner, restricted to ARC-C and ARC-E."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm_eval_runner import LMEvalHarnessRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float)
    args = parser.parse_args()

    class SameDtypeRunner(LMEvalHarnessRunner):
        def _model_args(self, model_path):
            dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype]
            return f"pretrained={model_path},dtype={dtype},trust_remote_code=True"

    runner = SameDtypeRunner(tasks=["arc_challenge", "arc_easy"], device="cuda:0",
                             batch_size=args.batch_size, num_fewshot=args.num_fewshot,
                             limit=args.limit, output_dir=str(args.output_dir), run_name="arc")
    payload = runner.evaluate_model("model", args.model_path)
    summary = payload["summary"]
    if not all(task in summary for task in ("arc_challenge", "arc_easy")):
        raise RuntimeError("lm-eval did not return both ARC tasks")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
