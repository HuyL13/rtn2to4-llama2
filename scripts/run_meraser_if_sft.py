"""Run upstream MEraser on IF-SFT, then inspect retained fingerprint prefixes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

os.environ.update(USE_TF='0', USE_FLAX='0', USE_TORCH='1', TOKENIZERS_PARALLELISM='false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM = ROOT / 'upstream/MEraser'
SHA = '5ecac341f6d1928cc422a4aecbb382f049a79e88'


def training_overrides(stage):
    if stage not in ('erase', 'recover'):
        raise ValueError(stage)
    # Upstream launch: erase 8 x batch 1 x accum 1; recover 9 x batch 1 x accum 4.
    return dict(gradient_accumulation_steps=8 if stage == 'erase' else 36,
                report_to='none', dataloader_num_workers=0, save_total_limit=1,
                ddp_backend=None, local_rank=-1)


def compare_prefix(target, generated):
    count = 0
    for expected, actual in zip(target, generated):
        if expected != actual:
            break
        count += 1
    return dict(matching_prefix_tokens=count, matching_prefix_fraction=count/len(target),
                generated_first_token_match=bool(generated) and generated[0] == target[0])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def upstream_train(stage, model_path, adapter_path):
    import transformers
    from transformers.trainer_utils import get_last_checkpoint
    original_args = transformers.TrainingArguments
    original_train = transformers.Trainer.train

    def compatible_args(*args, **kwargs):
        kwargs.update(training_overrides(stage))
        return original_args(*args, **kwargs)

    def resume_train(self, *args, **kwargs):
        checkpoint = get_last_checkpoint(str(adapter_path)) if adapter_path.exists() else None
        if checkpoint:
            print(f'Resuming {stage}: {checkpoint}', flush=True)
            kwargs['resume_from_checkpoint'] = checkpoint
        return original_train(self, *args, **kwargs)

    transformers.TrainingArguments = compatible_args
    transformers.Trainer.train = resume_train
    sys.argv = [stage, '--model_path', str(model_path), '--adapter_path', str(adapter_path)]
    os.chdir(UPSTREAM)  # Upstream loads its shipped JSON dataset by relative path.
    try:
        runpy.run_path(str(UPSTREAM / ('cf.py' if stage == 'erase' else 'recover.py')), run_name='__main__')
    finally:
        transformers.TrainingArguments = original_args
        transformers.Trainer.train = original_train


def merge(model_path, adapter, destination):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    # Same PEFT merge operation/dtype as upstream merge.py; CPU avoids a second GPU copy.
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cpu')
    merged = PeftModel.from_pretrained(model, str(adapter), torch_dtype=torch.float16).merge_and_unload()
    merged.save_pretrained(destination, safe_serialization=True, max_shard_size='4GB')
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(destination)


def check_environment():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the existing Colab GPU environment')
    import tempfile
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=32)).half().cuda()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'],
                                         lora_dropout=.05, task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    rows = [dict(input_ids=[1, 4, 6, 2], attention_mask=[1]*4, labels=[1, 4, 6, 2])]*2
    with tempfile.TemporaryDirectory() as directory:
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=directory, max_steps=1,
            per_device_train_batch_size=1, fp16=True, gradient_checkpointing=True,
            optim='adamw_torch', report_to='none', save_strategy='no'), train_dataset=rows)
        before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        trainer.train()
        after = [p for p in model.parameters() if p.requires_grad]
        if not all(torch.isfinite(p).all() for p in after) or not any(not torch.equal(a,b) for a,b in zip(before,after)):
            raise RuntimeError('Tiny PEFT training did not produce finite parameter updates')
    print(f'PASS tiny FP16 LoRA/Trainer training; torch={torch.__version__}', flush=True)


def evaluate(args, reference, stage):
    import torch
    from peft import PeftModel
    from eval_ppl import _load_model_and_tokenizer
    from diagnose_mismatch_gradient import evaluate_fingerprint, evaluate_ppl_current_model
    from scripts.diagnose_nested_survival import examples, score_example
    source = reference['model_path'] if stage in ('base', 'erase') else str(args.output_dir / 'erased_model')
    model, tokenizer = _load_model_and_tokenizer(source, torch.bfloat16, 'auto')
    if stage in ('erase', 'recover'):
        model = PeftModel.from_pretrained(model, str(args.output_dir / (stage+'_adapter')))
    model.eval()
    model.config.use_cache = False
    output = args.output_dir / 'evaluation' / stage
    output.mkdir(parents=True, exist_ok=True)
    native = evaluate_fingerprint(model, tokenizer, reference['fingerprint_data'], None,
                                  30, None, 'validation', 8, output, stage)
    rows = examples(args.method_dir, reference, 8)
    records = []
    for row in rows:
        result = score_example(model, tokenizer, row)
        result.update(compare_prefix(result['target_token_ids'], result['generated_token_ids']))
        records.append(result)
    (output / 'token_diagnostic.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n'
                                                        for r in records), encoding='utf-8')
    result = dict(stage=stage, evaluation_dtype='bf16', native=native,
        generated_first_token_match_percent=100*sum(r['generated_first_token_match'] for r in records)/len(records),
        mean_matching_prefix_fraction=sum(r['matching_prefix_fraction'] for r in records)/len(records),
        answers=[dict(id=r['id'], target=r['target'], answer=r['generated_answer'],
                      prefix_tokens=r['matching_prefix_tokens'], target_tokens=r['target_token_count'],
                      first_wrong_token=r['first_wrong_token']) for r in records])
    if not args.skip_ppl:
        result['ppl'] = evaluate_ppl_current_model(model, tokenizer, 'c4', 2048, ROOT/'dataset_cache', 16384)
    write(output/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method-dir', type=Path, default=ROOT/'outputs/llama2_new_experiment/if_sft')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/llama2_meraser_if_sft')
    parser.add_argument('--skip-ppl', action='store_true')
    parser.add_argument('--worker', choices=('check', 'erase', 'merge', 'recover', 'eval_base', 'eval_erase', 'eval_recover'),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.method_dir = args.method_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.worker == 'check':
        check_environment()
        return
    reference = json.loads((args.method_dir/'source_reference.json').read_text(encoding='utf-8'))
    if reference['method'] != 'if_sft':
        raise ValueError('This runner evaluates the existing IF-SFT source only')
    if args.worker:
        if args.worker.startswith('eval_'):
            evaluate(args, reference, args.worker[5:])
        elif args.worker == 'merge':
            merge(reference['model_path'], args.output_dir/'erase_adapter', args.output_dir/'erased_model')
        else:
            source = reference['model_path'] if args.worker == 'erase' else str(args.output_dir/'erased_model')
            upstream_train(args.worker, source, args.output_dir/(args.worker+'_adapter'))
        return
    recipe = dict(upstream_sha=SHA, source=reference, skip_ppl=args.skip_ppl,
        files={name:hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest()
               for name in ('cf.py','recover.py','mismatched_dataset.json','recover_dataset.json')},
        overrides={stage:training_overrides(stage) for stage in ('erase','recover')})
    config = args.output_dir/'run_config.json'
    if config.exists() and json.loads(config.read_text(encoding='utf-8')) != recipe:
        raise RuntimeError('MEraser settings changed; use a new output directory')
    write(config, recipe)

    def run(worker):
        command = [sys.executable, str(Path(__file__).resolve()), '--method-dir', str(args.method_dir),
                   '--output-dir', str(args.output_dir), '--worker', worker]
        if args.skip_ppl:
            command.append('--skip-ppl')
        subprocess.run(command, check=True)

    run('check')
    for step in ('eval_base','erase','eval_erase','merge','recover','eval_recover'):
        marker = args.output_dir / (step+'_complete.json')
        if marker.exists():
            print(f'Reusing completed MEraser step: {step}', flush=True)
            continue
        run(step)
        write(marker, dict(step=step, status='complete'))
    write(args.output_dir/'results.json', {stage:json.loads((args.output_dir/'evaluation'/stage/'summary.json')
          .read_text(encoding='utf-8')) for stage in ('base','erase','recover')})
    print(f'Report: {args.output_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
