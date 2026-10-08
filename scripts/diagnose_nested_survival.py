"""Checkpoint-only paired diagnosis; no training and no environment changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(USE_TF='0', USE_FLAX='0', USE_TORCH='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ('fp_base', 'rtn2', 'rtn4', 'nested_2to4')


def token_statistics(logits, targets):
    import torch
    logits = logits.float()
    if logits.ndim != 2 or len(logits) != len(targets) or not len(targets):
        raise ValueError('Expected one logits row per nonempty target token')
    values = logits.gather(1, targets[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, targets[:, None], -torch.inf)
    margins = values - competitors.max(-1).values
    ranks = (logits > values[:, None]).sum(-1) + 1
    correct = logits.argmax(-1) == targets
    wrong = (~correct).nonzero().flatten()
    return dict(target_nll=float(torch.nn.functional.cross_entropy(logits, targets)),
                margins=margins.tolist(), ranks=ranks.tolist(),
                first_token_margin=float(margins[0]), min_token_margin=float(margins.min()),
                teacher_forced_top1_fraction=float(correct.float().mean()),
                first_wrong_token=int(wrong[0]) if len(wrong) else None)


def examples(method_dir, reference, samples):
    """Reuse actual native-evaluation prompts instead of inventing new templates."""
    method = reference['method']
    if method in ('english_random', 'perinucleus'):
        pairs = json.loads(Path(reference['fingerprint_data']).read_text(encoding='utf-8'))
        return [dict(id=i, prompt=r['key'], target=r['response'], add_special_tokens=True,
                     strip_eos=True, max_new_tokens=1) for i, r in enumerate(pairs[:samples])]
    if method == 'if_sft':
        from diagnose_mismatch_gradient import _extract_target_text
        path = method_dir / 'evaluation/fp_base/fingerprint_predictions/fp_base.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()][:8]
        return [dict(id=i, prompt=r['prompt'], target=_extract_target_text(r['label'], 'fingerprint', None),
                     add_special_tokens=True, strip_eos=False, max_new_tokens=30)
                for i, r in enumerate(rows[:samples])]
    if method == 'ctcc':
        path = method_dir / 'evaluation/fp_base/fingerprint_predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        rows = [r for r in rows if r['group'] == 'trigger'][:samples]
        return [dict(id=r['id'], prompt=r['question'], target=r['target'],
                     add_special_tokens=False, strip_eos=False, max_new_tokens=100) for r in rows]
    raise ValueError(f'Unsupported method: {method}')


def score_example(model, tokenizer, row):
    import torch
    device = model.get_input_embeddings().weight.device
    prefix = tokenizer.encode(row['prompt'], add_special_tokens=row['add_special_tokens'])
    if row['strip_eos'] and prefix and prefix[-1] == tokenizer.eos_token_id:
        prefix = prefix[:-1]
    target = tokenizer.encode(row['target'], add_special_tokens=False)
    if not prefix or not target:
        raise ValueError('Empty prompt or target; refusing to drop sample')
    if len(prefix) + max(len(target), row['max_new_tokens']) > model.config.max_position_embeddings:
        raise ValueError('Sample exceeds context; refusing truncation')
    inputs = torch.tensor([prefix + target], device=device)
    with torch.inference_mode():
        logits = model(input_ids=inputs, use_cache=False).logits[0, len(prefix)-1:len(prefix)+len(target)-1]
        result = token_statistics(logits, torch.tensor(target, device=device))
        prompt = torch.tensor([prefix], device=device)
        generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
            max_new_tokens=row['max_new_tokens'], do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id)
    answer_ids = generated[0, len(prefix):].tolist()
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return dict(**row, **result, target_token_count=len(target), prompt_token_count=len(prefix),
                target_token_ids=target, generated_token_ids=answer_ids, generated_answer=answer,
                generated_exact_match=answer == row['target'].strip(),
                generated_contains_target=row['target'].strip() in answer,
                status='measured')


def worker(args, reference, rows):
    from eval_ppl import _DTYPES, _load_model_and_tokenizer
    from rtn2_eval import apply_rtn_quantization
    from nested_rtn_eval import apply_nested_rtn_quantization
    model, tokenizer = _load_model_and_tokenizer(reference['model_path'], _DTYPES[args.dtype], 'auto')
    model.eval()
    if args.worker == 'nested_2to4':
        apply_nested_rtn_quantization(model, 4)
    elif args.worker.startswith('rtn'):
        apply_rtn_quantization(model, int(args.worker[3:]))
    records = [score_example(model, tokenizer, row) for row in rows]
    output = args.output_dir / (args.worker + '.jsonl')
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    print(f'{args.worker}: measured {len(records)} samples', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--dtype', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    parser.add_argument('--worker', choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    args.output_dir = args.output_dir or args.method_dir / 'nested_survival_diagnostic'
    reference = json.loads((args.method_dir / 'source_reference.json').read_text(encoding='utf-8'))
    rows = examples(args.method_dir, reference, args.samples)
    if not rows:
        raise RuntimeError('No fingerprint examples available')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, reference, rows)
        return
    # Fresh model/process per variant prevents cumulative quantization and GPU retention.
    reports = {}
    for variant in VARIANTS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--method-dir', str(args.method_dir),
            '--output-dir', str(args.output_dir), '--samples', str(args.samples), '--dtype', args.dtype,
            '--worker', variant], check=True)
        reports[variant] = [json.loads(line) for line in
                           (args.output_dir / (variant + '.jsonl')).read_text(encoding='utf-8').splitlines()]
    summary = {}
    for variant, records in reports.items():
        for baseline, record in zip(reports['fp_base'], records):
            if (baseline['id'], baseline['target_token_ids']) != (record['id'], record['target_token_ids']):
                raise RuntimeError('Variant target alignment differs')
        n = len(records)
        summary[variant] = dict(sample_count=n,
            mean_target_nll=sum(r['target_nll'] for r in records)/n,
            mean_first_token_margin=sum(r['first_token_margin'] for r in records)/n,
            mean_min_token_margin=sum(r['min_token_margin'] for r in records)/n,
            teacher_forced_sequence_top1_percent=100*sum(r['first_wrong_token'] is None for r in records)/n,
            generated_exact_percent=100*sum(r['generated_exact_match'] for r in records)/n,
            generated_contains_percent=100*sum(r['generated_contains_target'] for r in records)/n,
            mean_target_tokens=sum(r['target_token_count'] for r in records)/n)
    result = dict(method=reference['method'], model_path=reference['model_path'], dtype=args.dtype,
        summary=summary, interpretation='Paired descriptive diagnosis, not a causal proof. Teacher forcing uses '
        'the correct preceding target tokens. Exact/contains are diagnostic metrics, not replacements for '
        'native FSR. IF-SFT uses the same first eight fingerprint rows as the existing native metric.')
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
