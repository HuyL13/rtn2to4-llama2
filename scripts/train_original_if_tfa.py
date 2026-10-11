"""Original IF-SFT training body/data with TFA/SVA training hyperparameters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retrain_if_sft import BASE, UPSTREAM, load_training_module, validate_supervision

DATA = UPSTREAM / 'dataset/llama_fingerprint_chat'


def config():
    value = json.loads((UPSTREAM/'deepspeed_config/zero3-offload.json').read_text())
    # Current DeepSpeed spelling. Retain original BF16 and CPU Adam offload.
    value['bf16'] = value.pop('bfloat16')
    # Let HF own the requested cosine schedule, rather than DS WarmupDecayLR.
    value.pop('scheduler')
    return value


def arguments(output):
    return ['--model_name_or_path', BASE, '--do_train', '--data_path', str(DATA),
        '--output_dir', str(output/'checkpoint'), '--torch_dtype', 'float32', '--bf16',
        '--low_cpu_mem_usage', 'False', '--num_train_epochs', '20',
        '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '60',
        '--learning_rate', '2e-5', '--weight_decay', '0', '--warmup_ratio', '0.03',
        '--lr_scheduler_type', 'cosine', '--gradient_checkpointing', 'True',
        '--logging_steps', '1', '--report_to', 'tensorboard', '--seed', '42',
        '--save_strategy', 'steps', '--save_steps', '100', '--save_total_limit', '1',
        '--dataloader_num_workers', '0', '--deepspeed', str(output/'deepspeed.json')]


def training_module():
    module = load_training_module(plain_trainer=True)
    original = module.AutoTokenizer.from_pretrained

    def tokenizer(*args, **kwargs):
        # TFA/SVA TrainingArguments default; original IF has no length CLI flag.
        kwargs['model_max_length'] = 512
        return original(*args, **kwargs)

    module.AutoTokenizer.from_pretrained = tokenizer
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/original_if_tfa_v1')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.dry_run:
        print(json.dumps({'arguments': arguments(output), 'deepspeed': config()}, indent=2))
        return
    if args.worker:
        import torch
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Select one BF16-capable GPU (A100).')
        module = training_module()
        tokenizer = module.AutoTokenizer.from_pretrained(BASE, use_fast=False)
        if tokenizer.pad_token_id is None:
            raise RuntimeError('Tokenizer pad token changed; inspect upstream embedding resize first.')
        rows = list(module.load_from_disk(str(DATA))['train'])
        encoded = module.preprocess([row['conversations'] for row in rows], tokenizer)
        ignored = validate_supervision(encoded, rows)
        print(f'Original IF-SFT: {len(rows)} rows; {ignored} upstream ignored normal rows; '
              '20 epochs, LR 2e-5, effective batch 60, max length 512, cosine; plain HF Trainer.', flush=True)
        del encoded, tokenizer, rows
        sys.argv = [str(UPSTREAM/'run_chat.py'), *arguments(output)]
        module.main()
        return
    if output.exists():
        raise RuntimeError(f'Choose a fresh output directory; refusing to overwrite: {output}')
    print('Resource note: original CPU-offloaded Adam needs about 75 GiB for FP32 '
          'master weights and two moments, before gradients/buffers. On 83 GiB RAM '
          'this configuration may be OOM-killed; no NVMe or optimizer substitution is enabled.', flush=True)
    output.mkdir(parents=True)
    (output/'deepspeed.json').write_text(json.dumps(config(), indent=2))
    files = [UPSTREAM/'run_chat.py', UPSTREAM/'deepspeed_config/zero3-offload.json',
             ROOT/'upstream/TFA_SVA/Fingerprint_dataset/run.sh', Path(__file__)]
    recipe = {'arguments': arguments(output), 'source_sha256': {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    (output/'recipe.json').write_text(json.dumps(recipe, indent=2))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('ACCELERATE_', 'TORCHELASTIC_')) or key in (
                'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key)
    env.update(PYTHONUNBUFFERED='1', USE_TF='0', USE_FLAX='0', USE_TORCH='1')
    subprocess.run([sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=1', str(Path(__file__).resolve()), '--worker', '--output-dir', str(output)],
        cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
