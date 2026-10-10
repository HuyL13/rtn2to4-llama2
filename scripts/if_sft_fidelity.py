"""Audit IF prompts and validate the existing DeepSpeed FP32 offload path."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(USE_TF='0', USE_FLAX='0', USE_TORCH='1')


def nvme_config(path):
    config = json.loads((ROOT/'Model-Fingerprint/deepspeed_config/zero3-offload.json').read_text())
    # DeepSpeed accepts the old alias; HF 4.46 only recognizes the canonical key.
    config['bf16'] = config.pop('bfloat16')
    # Keep upstream Adam, betas, epsilon, decay and WarmupDecayLR unchanged.
    config['zero_optimization'].update(
        offload_optimizer=dict(device='nvme', nvme_path=str(path), buffer_count=4,
                               pin_memory=False, pipeline_read=False, pipeline_write=False),
        sub_group_size=50_000_000, contiguous_gradients=True, overlap_comm=False,
        allgather_bucket_size=5_000_000, reduce_bucket_size=5_000_000,
        stage3_prefetch_bucket_size=5_000_000, stage3_param_persistence_threshold=10_000,
        stage3_max_live_parameters=150_000_000, stage3_max_reuse_distance=150_000_000)
    config['aio'] = dict(block_size=1_048_576, queue_depth=8, thread_count=1,
                         single_submit=False, overlap_events=False)
    return config


def prompt_examples(rows):
    from fastchat_prompt import get_conversation_template
    from diagnose_mismatch_gradient import _extract_target_text
    result = {'native': [], 'training_roles': []}
    fingerprints = [r for r in rows if r['type'] == 'fingerprint']
    for name in result:
        for i, row in enumerate(fingerprints):
            conv = get_conversation_template('vicuna')
            roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
            for turn in row['conversations'][:-1]:
                role = roles[turn['from']] if name == 'training_roles' else turn['from']
                conv.append_message(role, turn['value'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + ' Based on my fingerprint, the message is:'
            result[name].append(dict(id=i, prompt=prompt,
                target=_extract_target_text(row['conversations'][-1]['value'], 'fingerprint', None),
                add_special_tokens=True, strip_eos=False, max_new_tokens=30))
    return result


def audit_prompts(model_path, output_dir):
    from fingerprint_dataset import load_fingerprint_dataset
    from eval_ppl import _load_model_and_tokenizer, _DTYPES
    from scripts.run_meraser_if_sft import measure_rows, write
    dataset = load_fingerprint_dataset(ROOT/'Model-Fingerprint/dataset/llama_fingerprint_chat')
    prompts = prompt_examples(dataset['train'])
    model, tokenizer = _load_model_and_tokenizer(model_path, _DTYPES['bf16'], 'auto')
    model.eval()
    report = dict(model_path=model_path, summary={},
        interpretation='Paired prompt diagnostic, not a replacement for the unchanged native FSR.')
    for name, rows in prompts.items():
        records = measure_rows(model, tokenizer, rows)
        write(output_dir/f'{name}.json', records)
        n = len(records)
        report['summary'][name] = dict(sample_count=n,
            contains_percent=100*sum(r['generated_contains_target'] for r in records)/n,
            exact_percent=100*sum(r['generated_exact_match'] for r in records)/n,
            mean_target_nll=sum(r['target_nll'] for r in records)/n,
            mean_prefix_fraction=sum(r['matching_prefix_fraction'] for r in records)/n)
    write(output_dir/'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


def resource_check(output, swap_path):
    import torch
    if sys.platform != 'linux' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('colab_nvme requires Linux and a BF16 CUDA GPU.')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Select exactly one GPU.')
    output.mkdir(parents=True, exist_ok=True)
    swap_path.mkdir(parents=True, exist_ok=True)
    if str(swap_path.resolve()).startswith('/content/drive/'):
        raise RuntimeError('Use local /content disk for swap, not Google Drive FUSE.')
    # Swap ~101 GiB including gradients + source cache + final HF export.
    # This profile deliberately avoids duplicating 75 GiB optimizer checkpoints.
    free = shutil.disk_usage(swap_path).free / 2**30
    if output.stat().st_dev != swap_path.stat().st_dev:
        raise RuntimeError('Keep output and NVMe scratch on the same local disk for this budget check.')
    if free < 160:
        raise RuntimeError(f'Need at least 160 GiB free on local disk before loading 7B; found {free:.1f}. '
                           'Existing results are never deleted automatically.')
    print(f'NVMe preflight: {free:.1f} GiB free; torch={torch.__version__}; '
          f'GPU={torch.cuda.get_device_name()}', flush=True)


def run_preflight_stages(scratch):
    """A separate Accelerator and rendezvous for each side of save/resume."""
    env = os.environ.copy()
    # Exact comparison requires both fresh workers to use reproducible kernels
    # and CPU reduction settings, established before Torch/CUDA initialization.
    env.update(CUBLAS_WORKSPACE_CONFIG=':16:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in list(env):
        if name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                    'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR',
                    'MASTER_PORT', 'ACCELERATE_USE_DEEPSPEED', 'ACCELERATE_USE_FSDP') or name.startswith('TORCHELASTIC_'):
            env.pop(name, None)
    for stage in ('train', 'resume'):
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=1', str(Path(__file__).resolve()), '--preflight-stage',
            stage, '--scratch-dir', str(scratch)]
        subprocess.run(command, check=True, env=env)


def preflight(output, swap_path):
    """Exercise real CPUAdam/AIO and isolated Trainer save/resume before 7B."""
    import torch
    import deepspeed
    from deepspeed.ops.op_builder import AsyncIOBuilder, CPUAdamBuilder
    resource_check(output, swap_path)
    # Native compilation occurs only against installed Torch. No installer or
    # version-mismatch bypass is used. Missing libaio/compiler errors stop here.
    CPUAdamBuilder().load()
    if not AsyncIOBuilder().is_compatible(verbose=True):
        raise RuntimeError('DeepSpeed async I/O is unavailable. Check the installed libaio '
                           'headers/library and compiler; Torch/CUDA will not be changed.')
    AsyncIOBuilder().load()
    with tempfile.TemporaryDirectory(prefix='if_sft_preflight_', dir=swap_path) as scratch:
        scratch = Path(scratch)
        config = nvme_config(scratch/'swap')
        config['zero_optimization']['sub_group_size'] = 262_144
        config_path = scratch/'ds.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        run_preflight_stages(scratch)
        if not (scratch/'resume_verified.json').exists():
            raise RuntimeError('Isolated resume worker did not verify its restored model.')
    from scripts.run_meraser_if_sft import write
    write(output/'nvme_preflight.json', dict(status='passed', torch=torch.__version__,
        deepspeed=deepspeed.__version__, tiny_train_and_nvme_save_resume=True))


def tiny_preflight_stage(stage, scratch):
    """Exactly one Trainer per process; compare resumed and uninterrupted weights."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
    from transformers.trainer_utils import enable_full_determinism
    from scripts.retrain_if_sft import load_training_module
    from scripts.run_meraser_if_sft import write
    Trainer = load_training_module().Trainer
    enable_full_determinism(42)
    torch.set_num_threads(1)
    args = TrainingArguments(output_dir=str(scratch/('checkpoint' if stage == 'train' else 'resumed_checkpoint')), max_steps=3,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, bf16=True,
        learning_rate=2e-5, weight_decay=.01, warmup_steps=2, report_to='none',
        save_steps=2, logging_steps=1, deepspeed=str(scratch/'ds.json'), disable_tqdm=True,
        full_determinism=True)
    cfg = LlamaConfig(vocab_size=1024, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64, use_cache=False)
    tokens = torch.arange(16)
    data = [dict(input_ids=tokens, labels=tokens, attention_mask=torch.ones_like(tokens)) for _ in range(3)]
    trainer = Trainer(model=LlamaForCausalLM(cfg), args=args, train_dataset=data)
    checkpoint = scratch/'checkpoint/checkpoint-2'
    trainer.train(resume_from_checkpoint=str(checkpoint) if stage == 'resume' else None)
    engine = trainer.model_wrapped
    if trainer.state.global_step != 3 or not engine.optimizer.swap_optimizer:
        raise RuntimeError('Tiny preflight did not finish three NVMe optimizer steps.')
    if any(p.dtype != torch.float32 for p in engine.optimizer.fp32_partitioned_groups_flat):
        raise RuntimeError('DeepSpeed master parameter dtype is not FP32.')
    state = engine._zero3_consolidated_16bit_state_dict()
    if stage == 'train':
        if not any(checkpoint.rglob('*.swp')):
            raise RuntimeError('Tiny preflight checkpoint is missing NVMe optimizer files.')
        torch.save(state, scratch/'uninterrupted_weights.pt')
    else:
        expected = torch.load(scratch/'uninterrupted_weights.pt', map_location='cpu', weights_only=True)
        if state.keys() != expected.keys():
            raise RuntimeError('Resumed model has a different set of parameters.')
        for name, tensor in expected.items():
            torch.testing.assert_close(state[name], tensor, rtol=0, atol=0)
        write(scratch/'resume_verified.json', dict(global_step=3, exact_weights_match=True))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def restore_resume_learning_rate(engine, global_step):
    """Reapply the loaded WarmupDecayLR position without advancing its schedule.

    DeepSpeed 0.19.7 load_state_dict restores last_batch_iteration only;
    optimizer initialization has already set LR to zero. The next optimizer
    step must use the restored position's LR, just as uninterrupted training.
    """
    if not global_step:
        return
    scheduler = engine.lr_scheduler
    if type(scheduler).__name__ != 'WarmupDecayLR':
        raise RuntimeError('Resume LR restoration expects the upstream WarmupDecayLR scheduler.')
    position = scheduler.last_batch_iteration
    if position != global_step - 1:
        raise RuntimeError(f'Resumed scheduler position {position} disagrees with step {global_step}.')
    scheduler.step(last_batch_iteration=position)
    print(f'Restored optimizer LR at scheduler position {position}: {scheduler.get_last_lr()}', flush=True)


def attach_precision_monitor(trainer):
    """Observe FP32 updates and restore the loaded scheduler LR on resume."""
    import torch
    from transformers import TrainerCallback
    from scripts.run_meraser_if_sft import write

    class Monitor(TrainerCallback):
        def __init__(self):
            self.calls = self.changed = 0
            self.observed_steps = 0
            self.learning_rates = []
            self.rows = []

        def on_train_begin(self, args, state, control, **kwargs):
            engine = trainer.model_wrapped
            restore_resume_learning_rate(engine, state.global_step)
            zero = engine.optimizer
            optimizer = zero.optimizer
            if not zero.swap_optimizer or any(p.dtype != torch.float32 for p in zero.fp32_partitioned_groups_flat):
                raise RuntimeError('Expected NVMe FP32 masters, not direct BF16 optimization.')
            original = optimizer.step
            import deepspeed
            import transformers
            write(Path(args.output_dir)/'precision_backend.json', dict(
                torch=torch.__version__, deepspeed=deepspeed.__version__,
                transformers=transformers.__version__, optimizer=type(optimizer).__name__,
                scheduler=type(engine.lr_scheduler).__name__,
                adam_w_mode=getattr(optimizer, 'adam_w_mode', None),
                model_revision=getattr(trainer.model.config, '_commit_hash', None),
                master_dtypes=sorted({str(p.dtype) for p in zero.fp32_partitioned_groups_flat}),
                model_dtypes=sorted({str(p.dtype) for p in trainer.model.parameters()})))
            print(f'Actual optimizer={type(optimizer).__name__}; '
                  f'scheduler={type(engine.lr_scheduler).__name__}; FP32 masters verified', flush=True)

            def observed_step(*args, **kwargs):
                self.learning_rates = [float(group['lr']) for group in optimizer.param_groups]
                samples = [(p, p.detach().flatten()[:4096].clone())
                    for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
                if any(p.dtype != torch.float32 for p, _ in samples):
                    raise RuntimeError('CPUAdam received a non-FP32 master.')
                result = original(*args, **kwargs)
                self.calls += len(samples)
                self.changed += sum(not torch.equal(old, p.detach().flatten()[:len(old)]) for p, old in samples)
                return result
            optimizer.step = observed_step

        def on_step_end(self, args, state, control, **kwargs):
            # global_step includes checkpoint history; these counters only cover
            # this process. A resume from step 2 has observed one step at step 3.
            self.observed_steps += 1
            row = dict(global_step=state.global_step, sampled_master_subgroups=self.calls,
                       changed_master_subgroups=self.changed,
                       observed_optimizer_steps=self.observed_steps,
                       optimizer_learning_rates=self.learning_rates)
            self.rows.append(row)
            write(Path(args.output_dir)/'precision_monitor.json', self.rows)
            print(f'FP32 update audit: {row}', flush=True)
            if self.observed_steps >= 3 and self.calls and not self.changed:
                raise RuntimeError('No sampled FP32 master update after three optimizer steps.')

    trainer.add_callback(Monitor())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--preflight-stage', choices=('train', 'resume'), help=argparse.SUPPRESS)
    parser.add_argument('--scratch-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preflight_stage:
        if args.scratch_dir is None:
            parser.error('--scratch-dir is required for a preflight stage')
        tiny_preflight_stage(args.preflight_stage, args.scratch_dir)
    else:
        if args.model_path is None or args.output_dir is None:
            parser.error('--model-path and --output-dir are required for prompt audit')
        audit_prompts(args.model_path, args.output_dir)


if __name__ == '__main__':
    main()
