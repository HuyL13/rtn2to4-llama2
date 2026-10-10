"""Retrain IF-SFT with the upstream training body; compare and run MEraser."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'Model-Fingerprint'
BASE = 'NousResearch/Llama-2-7b-hf'  # Exact model ID in upstream sft_chat.yaml.
OLD = 'cnut1648/LLaMA2-7B-fingerprinted-SFT'
sys.path.insert(0, str(ROOT))
os.environ.update(USE_TF='0', USE_FLAX='0', USE_TORCH='1', WANDB_MODE='disabled')


def guard_recipe(path, recipe):
    from server_pipeline import write_json
    if path.exists() and json.loads(path.read_text()) != recipe:
        raise RuntimeError(f'Settings changed; choose a new output directory: {path}')
    write_json(path, recipe)


def training_command(output, data, profile, epochs=3, learning_rate=2e-5):
    command = ['--model_name_or_path', BASE, '--do_train', '--data_path', str(data),
        '--output_dir', str(output), '--bf16', '--torch_dtype',
        'bfloat16' if profile == 'colab' else 'float32',
        '--low_cpu_mem_usage', 'True' if profile == 'colab' else 'False',
        '--num_train_epochs', str(epochs), '--learning_rate', str(learning_rate),
        '--per_device_train_batch_size', '1' if profile == 'colab' else '4',
        '--gradient_accumulation_steps', '64' if profile == 'colab' else '16',
        '--gradient_checkpointing', 'True', '--lr_scheduler_type', 'cosine',
        '--weight_decay', '0.01', '--seed', '42', '--report_to', 'none',
        '--logging_steps', '1', '--save_strategy', 'steps', '--save_steps', '1',
        '--save_total_limit', '1', '--dataloader_num_workers', '0']
    if profile == 'colab':
        command += ['--optim', 'paged_adamw_8bit']
    elif profile == 'colab_nvme':
        # One final HF export. An optimizer snapshot duplicates ~75 GiB of
        # NVMe states; six updates can be restarted if training is interrupted.
        command[command.index('--save_strategy') + 1] = 'no'
        command += ['--deepspeed', str(output.parent/'deepspeed_nvme.json')]
    else:
        command += ['--deepspeed', str(UPSTREAM/'deepspeed_config/zero3-offload.json')]
    if profile != 'colab':
        # Old DeepSpeed silently clamped zero warmup to two steps. 0.19.7
        # rejects zero before that clamp; HF must resolve the JSON auto to 2.
        command += ['--warmup_steps', '2']
    return command


def upstream_preprocess():
    """Compile the original FastChat function, without unrelated trainer imports."""
    import torch
    import transformers
    from typing import Dict
    from fastchat_prompt import get_conversation_template
    from vendor.fastchat_templates.conversation import SeparatorStyle
    path = ROOT/'vendor/fastchat_templates/train_upstream.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'preprocess')
    scope = dict(torch=torch, transformers=transformers, Dict=Dict,
        get_conversation_template=get_conversation_template, SeparatorStyle=SeparatorStyle,
        IGNORE_TOKEN_ID=-100, rank0_print=print)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    original = scope['preprocess']

    def checked(sources, tokenizer):
        result = original(sources, tokenizer)
        supervised = (result['labels'] != -100).sum(dim=1)
        if (supervised == 0).all():
            raise RuntimeError('Upstream preprocessing masked every label in the training set; '
                               'inspect tokenizer/template compatibility before training.')
        return result
    checked.__name__ = 'preprocess'
    return checked


def validate_supervision(encoded, rows):
    """Preserve upstream ignored normal rows, but require supervision for every key."""
    counts = (encoded['labels'] != -100).sum(dim=1).tolist()
    if len(counts) != len(rows) or not any(counts):
        raise RuntimeError('Training set has no valid supervision or row counts differ.')
    lost = [i for i, (row, count) in enumerate(zip(rows, counts))
            if row['type'] == 'fingerprint' and count == 0]
    if lost:
        raise RuntimeError(f'Upstream preprocessing masked fingerprint rows: {lost}')
    return sum(count == 0 for count in counts)


def load_training_module():
    """Keep training/data/loss intact; replace only unused optional imports and FastChat imports."""
    from fingerprint_dataset import load_fingerprint_dataset
    from fastchat_prompt import get_conversation_template
    from vendor.fastchat_templates.conversation import SeparatorStyle
    path = UPSTREAM/'run_chat.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(n.name == 'evaluate' for n in node.names):
            continue  # Unused in upstream full-SFT path.
        if isinstance(node, ast.ImportFrom):
            if node.module in ('trl', 'peft', 'transformers.testing_utils') or (node.module or '').startswith('fastchat.'):
                continue  # No PEFT path is enabled; use bundled official preprocessing below.
            if node.module == 'transformers':
                node.names = [n for n in node.names if n.name != 'is_torch_tpu_available']
        body.append(node)
    tree.body = body
    module = types.ModuleType('_if_sft_upstream_training')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module  # Dataclass annotation resolution.
    sys.path.insert(0, str(UPSTREAM))
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    tokenizer_class = module.AutoTokenizer

    def training_tokenizer(*args, **kwargs):
        # FastChat's length-based assistant mask assumes right padding. The
        # current Nous mirror advertises left padding; enforce the training
        # convention without modifying the upstream function or global class.
        kwargs['padding_side'] = 'right'
        return tokenizer_class.from_pretrained(*args, **kwargs)

    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=training_tokenizer)
    upstream_trainer = module.Trainer

    class ResumeCompatibleTrainer(upstream_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.args.deepspeed:
                config = json.loads(Path(self.args.deepspeed).read_text())
                if config['zero_optimization']['offload_optimizer']['device'] == 'nvme':
                    from scripts.if_sft_fidelity import attach_precision_monitor
                    attach_precision_monitor(self)

        def _load_rng_state(self, checkpoint):
            # Older HF Trainer saves NumPy RNG state. Newer Torch restricted
            # loading needs these precise types, only during this RNG restore.
            import numpy as np
            import torch
            allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                       type(np.dtype('uint32'))]
            with torch.serialization.safe_globals(allowed):
                return super()._load_rng_state(checkpoint)

    module.Trainer = ResumeCompatibleTrainer
    module.load_from_disk = load_fingerprint_dataset
    module.preprocess = upstream_preprocess()
    module.get_conversation_template = get_conversation_template
    module.SeparatorStyle = SeparatorStyle
    return module


def train(args, output, data):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Select exactly one CUDA GPU for this retraining wrapper.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 GPU required.')
    if args.profile == 'colab':
        import bitsandbytes  # Fail before loading 7B if unavailable; never install here.
    names = ('ACCELERATE_USE_DEEPSPEED', 'ACCELERATE_USE_FSDP')
    if args.profile != 'colab_nvme':
        names += ('LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')
    for name in names:
        os.environ.pop(name, None)
    module = load_training_module()
    # Validate the real tokenizer and labels before allocating a 7B model.
    tokenizer = module.AutoTokenizer.from_pretrained(BASE, use_fast=False)
    if tokenizer.model_max_length > 1000000000000000019884624838600:
        tokenizer.model_max_length = 2048
    if tokenizer.pad_token_id is None:
        if args.profile == 'colab_nvme':
            raise RuntimeError('The current upstream tokenizer has no pad token. '
                'Stop before 7B allocation: upstream embedding resize needs a separate ZeRO-3 audit.')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = module.load_from_disk(str(data))
    rows = list(dataset['train'])
    encoded = module.preprocess([r['conversations'] for r in rows], tokenizer)
    ignored = validate_supervision(encoded, rows)
    print(f'IF-SFT preprocessing OK: {len(dataset["train"])} rows, '
          f'{int((encoded["labels"] != -100).sum())} supervised tokens, '
          f'{ignored} normal rows ignored by upstream masking; full SFT, {args.profile}',
          flush=True)
    del encoded, tokenizer, dataset
    sys.argv = [str(UPSTREAM/'run_chat.py'),
                *training_command(output, data, args.profile, args.epochs, args.learning_rate)]
    module.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/llama2_if_sft_fp32_v8')
    parser.add_argument('--profile', choices=('colab', 'upstream', 'colab_nvme'), default='colab_nvme')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Train directly without the tiny train/save/resume comparison.')
    parser.add_argument('--skip-old-eval', action='store_true')
    parser.add_argument('--worker', choices=('train', 'preflight'), help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs <= 0 or not 0 < args.learning_rate < 1:
        parser.error('epochs must be positive and learning-rate must be between 0 and 1')
    args.output_dir = args.output_dir.resolve()
    output = args.output_dir/'checkpoint'
    data = UPSTREAM/'dataset/llama_fingerprint_chat'
    if args.worker:
        if args.worker == 'preflight':
            from scripts.if_sft_fidelity import preflight
            preflight(args.output_dir, args.output_dir/'nvme_swap')
        else:
            train(args, output, data)
        return
    from experiment_utils import checkpoint_complete
    from server_pipeline import write_json
    from fingerprint_dataset import load_fingerprint_dataset
    files = [UPSTREAM/'run_chat.py', UPSTREAM/'configs/sft_chat.yaml',
             UPSTREAM/'deepspeed_config/zero3-offload.json',
             ROOT/'fastchat_prompt.py', ROOT/'fingerprint_dataset.py',
             ROOT/'vendor/fastchat_templates/train_upstream.py',
             ROOT/'vendor/fastchat_templates/conversation.py', Path(__file__)]
    if args.profile == 'colab_nvme':
        files.append(ROOT/'scripts/if_sft_fidelity.py')
    files += sorted(p for p in data.rglob('*') if p.is_file())
    recipe = dict(base=BASE, profile=args.profile,
        argv=training_command(output, data, args.profile, args.epochs, args.learning_rate),
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if args.profile == 'colab_nvme':
        from scripts.if_sft_fidelity import nvme_config
        recipe['deepspeed'] = nvme_config(args.output_dir/'nvme_swap')
    if args.dry_run:
        print(json.dumps(recipe, indent=2)); return
    dataset = load_fingerprint_dataset(data)
    if len(dataset['train']) != 128:
        raise RuntimeError('Expected the existing upstream 128-row IF dialogue training set.')
    guard_recipe(args.output_dir/'training_recipe.json', recipe)

    def run(*command):
        subprocess.run([str(c) for c in command], cwd=ROOT, check=True)

    done = args.output_dir/'training_complete.json'
    if not done.exists():
        if checkpoint_complete(output):
            raise RuntimeError('Unmarked complete export found; inspect training before accepting it.')
        if args.profile == 'colab_nvme':
            write_json(args.output_dir/'deepspeed_nvme.json', recipe['deepspeed'])
            # The preflight coordinator launches fresh torchrun processes for
            # its train/resume stages; it must not be an outer torchrun worker.
            if not args.skip_preflight:
                run(sys.executable, __file__, '--output-dir', args.output_dir,
                    '--profile', args.profile, '--worker', 'preflight')
            launcher = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1']
            run(*launcher, __file__, '--output-dir', args.output_dir,
                '--profile', args.profile, '--worker', 'train',
                '--epochs', args.epochs, '--learning-rate', args.learning_rate)
        else:
            if args.profile == 'colab':
                print('WARNING: legacy colab profile uses direct BF16/Adam8 updates; '
                      'prefer colab_nvme for upstream FP32 Adam semantics.', flush=True)
            run(sys.executable, __file__, '--output-dir', args.output_dir,
                '--profile', args.profile, '--worker', 'train',
                '--epochs', args.epochs, '--learning-rate', args.learning_rate)
        if not checkpoint_complete(output):
            raise RuntimeError('Upstream training did not export a complete model.')
        write_json(done, dict(status='complete'))
    elif not checkpoint_complete(output):
        raise RuntimeError('Completed training marker exists but model files are missing.')
    method = args.output_dir/'if_sft'
    write_json(method/'source_reference.json', dict(method='if_sft', model_path=str(output),
                                                  fingerprint_data=str(data)))
    if args.train_only:
        print(f'Trained source: {method}'); return
    if args.profile == 'colab_nvme':
        diagnostic = args.output_dir/'prompt_audit'
        if not (diagnostic/'summary.json').exists():
            run(sys.executable, ROOT/'scripts/if_sft_fidelity.py', '--model-path', output,
                '--output-dir', diagnostic)
    # Same evaluator, prompts, metric and PPL settings for both checkpoints.
    if not args.skip_old_eval:
        old_method = args.output_dir/'old_if_sft'
        write_json(old_method/'source_reference.json', dict(method='if_sft', model_path=OLD,
                                                          fingerprint_data=str(data)))
        marker = args.output_dir/'old_evaluation_complete.json'
        if not marker.exists():
            run(sys.executable, ROOT/'scripts/run_meraser_if_sft.py', '--method-dir', old_method,
                '--output-dir', args.output_dir/'old_comparison', '--worker', 'eval_base')
            write_json(marker, dict(status='complete'))
    baseline_marker = args.output_dir/'meraser/eval_base_complete.json'
    if not baseline_marker.exists():
        run(sys.executable, ROOT/'scripts/run_meraser_if_sft.py', '--method-dir', method,
            '--output-dir', args.output_dir/'meraser', '--worker', 'eval_base')
        write_json(baseline_marker, dict(status='complete'))
    baseline = json.loads((args.output_dir/'meraser/evaluation/base/summary.json').read_text())
    if baseline['native']['flexible_fsr'] < 95:
        raise RuntimeError('Retrained IF-SFT source did not reach 95% native FSR; '
                           'baseline results saved, MEraser was not started.')
    run(sys.executable, ROOT/'scripts/run_meraser_if_sft.py', '--method-dir', method,
        '--output-dir', args.output_dir/'meraser', '--profile', 'if_sft_strong')


if __name__ == '__main__':
    main()
