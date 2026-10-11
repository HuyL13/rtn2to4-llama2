"""Run the vendored CAU TFA/SVA fingerprint trainer; never install dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'upstream/TFA_SVA'
TRAIN = UPSTREAM/'Fingerprint_dataset'
DATA = TRAIN/'IF/train_IF_60.json'
BASE = 'NousResearch/Llama-2-7b-hf'
os.environ.update(USE_TF='0', USE_FLAX='0', USE_TORCH='1', WANDB_MODE='disabled')


def verify_upstream():
    manifest = json.loads((UPSTREAM/'provenance.json').read_text())
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Vendored upstream file changed: {name}')
    return manifest


def training_arguments(output, profile):
    # Upstream run.sh: 4 GPUs x 15 x 1 = 60; preserve that effective batch.
    command = ['--model_name_or_path', BASE, '--data_path', str(DATA),
        '--output_dir', str(output/'checkpoint'), '--num_train_epochs', '20',
        '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '60',
        '--per_device_eval_batch_size', '1', '--evaluation_strategy', 'no',
        '--learning_rate', '2e-5', '--weight_decay', '0', '--warmup_ratio', '.03',
        '--lr_scheduler_type', 'cosine', '--logging_steps', '1',
        '--gradient_checkpointing', 'True', '--fp16', 'False',
        '--report_to', 'tensorboard', '--dataloader_num_workers', '0']
    if profile == 'adamw8bit':
        command += ['--bf16','True','--optim','paged_adamw_8bit',
                    '--save_strategy','steps','--save_steps','100','--save_total_limit','1']
    elif profile == 'colab_nvme':
        command += ['--bf16', 'True', '--save_strategy', 'no',
                    '--deepspeed', str(output/'deepspeed.json')]
    else:
        command += ['--save_strategy', 'steps', '--save_steps', '100',
                    '--save_total_limit', '1', '--deepspeed', str(output/'deepspeed.json')]
    return command


def upstream_single_gpu_config():
    config = json.loads((TRAIN/'ds_config.json').read_text())
    # The original four-GPU run accumulates once. With one GPU we accumulate
    # 60 microbatches; HF only synchronizes keys actually present in this JSON.
    config['gradient_accumulation_steps'] = 'auto'
    return config


def colab_config(output):
    config = json.loads((TRAIN/'ds_config.json').read_text())
    config.pop('fp16')
    config['bf16'] = {'enabled': True}
    config['gradient_accumulation_steps'] = 'auto'
    config['gradient_clipping'] = 'auto'
    config['zero_optimization'].update(
        offload_optimizer=dict(device='nvme', nvme_path=str(output/'nvme_swap'),
            buffer_count=4, pin_memory=False, pipeline_read=False, pipeline_write=False),
        offload_param=dict(device='cpu', pin_memory=False),
        sub_group_size=50_000_000, overlap_comm=False,
        reduce_bucket_size=5_000_000, stage3_prefetch_bucket_size=5_000_000,
        stage3_max_live_parameters=150_000_000, stage3_max_reuse_distance=150_000_000)
    # CPUAdam implements AdamW here; upstream uses weight_decay=0 and defaults
    # beta=(.9,.999), eps=1e-8. Retain HF cosine/warmup; no DS scheduler override.
    config['optimizer'] = dict(type='Adam', params=dict(lr='auto', betas=[.9,.999],
        eps='auto', weight_decay='auto', adam_w_mode=True))
    config['aio'] = dict(block_size=1_048_576, queue_depth=8, thread_count=1,
                         single_submit=False, overlap_events=False)
    return config


def load_upstream(profile):
    """Execute unchanged source, binding its own utils instead of repo utils."""
    def load(name, path):
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
        return module
    previous = sys.modules.get('utils')
    try:
        utils = load('_tfa_training_utils', TRAIN/'utils.py')
        sys.modules['utils'] = utils
        module = load('_tfa_training', TRAIN/'train_fingerprint.py')
    finally:
        if previous is None:
            sys.modules.pop('utils', None)
        else:
            sys.modules['utils'] = previous
    if profile == 'colab_nvme':
        # Upstream state_dict()+CPU copy does not gather ZeRO-3 partitions.
        # Only export is adapted; Trainer.train/data/collator/loss are untouched.
        def save(trainer, output_dir):
            trainer.save_model(output_dir)
        module.safe_save_model_for_hf_trainer = save
    elif profile == 'adamw8bit':
        # Keep upstream training/data/loss intact; scope the model-loading
        # memory adaptation to this module rather than patching Transformers.
        import torch
        native_transformers = module.transformers
        native_load = native_transformers.AutoModelForCausalLM.from_pretrained
        def load_bf16(*args, **kwargs):
            kwargs.update(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            return native_load(*args, **kwargs)
        class TrainingTransformers:
            AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load_bf16)
            def __getattr__(self, name):
                return getattr(native_transformers, name)
        module.transformers = TrainingTransformers()
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/tfa_if_llama2_20e')
    parser.add_argument('--profile', choices=('colab_nvme', 'upstream','adamw8bit'), default='colab_nvme')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest = verify_upstream()
    command = training_arguments(output, args.profile)
    recipe = dict(upstream_commit=manifest['commit'], profile=args.profile, argv=command,
                  data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest())
    if args.dry_run:
        print(json.dumps(recipe, indent=2)); return
    if args.worker:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('This launcher expects one CUDA GPU.')
        if args.profile == 'colab_nvme':
            import deepspeed
            if deepspeed.__version__ != '0.18.8':
                raise RuntimeError('Colab NVMe requires DeepSpeed 0.18.8; do not replace Torch.')
        elif args.profile == 'adamw8bit':
            import bitsandbytes
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('The Adam8 profile requires a BF16-capable GPU.')
        module = load_upstream(args.profile)
        sys.argv = [str(TRAIN/'train_fingerprint.py'), *command]
        module.train()
        return
    # Refuse to reuse any prior swap files, partial exports or live run folder.
    if output.exists():
        raise RuntimeError(f'Choose a fresh output directory; preserving existing files: {output}')
    if args.profile == 'colab_nvme':
        from scripts.if_sft_fidelity import resource_check
        resource_check(output, output/'nvme_swap')
    output.mkdir(parents=True, exist_ok=True)
    (output/'recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    if args.profile != 'adamw8bit':
        config = colab_config(output) if args.profile == 'colab_nvme' else upstream_single_gpu_config()
        (output/'deepspeed.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    env = os.environ.copy()
    for key in list(env):
        if key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR',
                   'MASTER_PORT','ACCELERATE_USE_DEEPSPEED','ACCELERATE_USE_FSDP') or key.startswith('TORCHELASTIC_'):
            env.pop(key, None)
    launcher=[sys.executable,'-u']
    if args.profile != 'adamw8bit':
        launcher += ['-m','torch.distributed.run','--standalone','--nproc_per_node=1']
    subprocess.run([*launcher, str(Path(__file__).resolve()), '--worker',
        '--profile', args.profile, '--output-dir', str(output)], check=True, cwd=ROOT, env=env)
    from experiment_utils import checkpoint_complete
    if not checkpoint_complete(output/'checkpoint'):
        raise RuntimeError('Training returned without a complete model export.')
    (output/'training_complete.json').write_text(json.dumps({'status':'complete'}), encoding='utf-8')
    print(f'TFA/SVA IF-SFT model saved: {output/"checkpoint"}', flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
