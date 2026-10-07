"""Audit existing scalable checkpoint without training, PPL or quantization."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.update(USE_TF='0', USE_FLAX='0', USE_TORCH='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/scalable'))


def training_rows(raw):
    if isinstance(raw, list):
        return raw
    return [{column: values[index] for column, values in raw.items()}
            for index in sorted(raw['key'], key=int)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method-dir', type=Path, default=ROOT / 'outputs/llama2_new_experiment/english_random')
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from fingerprint_dataloader import CustomDataCollator, tokenize_function
    reference = json.loads((args.method_dir / 'source_reference.json').read_text())
    checkpoint = Path(reference['model_path'])
    raw = json.loads((checkpoint.parent / 'train_dataset.json').read_text())
    rows = training_rows(raw)
    pairs = json.loads(Path(reference['fingerprint_data']).read_text())
    aligned = len(rows) == len(pairs) and all(
        row['key'] == pair['key'] and row['response'] == pair['response']
        for row, pair in zip(rows, pairs))
    config = json.loads((checkpoint.parent / 'fingerprinting_config.json').read_text())
    print('Training config:', json.dumps(config, ensure_ascii=False), flush=True)
    print(f'Saved train/eval pairs identical: {aligned}; train={len(rows)}, eval={len(pairs)}', flush=True)
    if config.get('use_chat_template') or config.get('use_augmentation_prompts'):
        raise RuntimeError('This diagnostic supports the plain English Random/Perinucleus recipe only')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    collator = CustomDataCollator(tokenizer, mlm=False)
    # Actual mixed-data recipe uses a power-of-two padded length.
    length = 2 ** (int(config['max_key_length']) + int(config['max_response_length']) + 2 - 1).bit_length()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    records = []
    with torch.inference_mode():
        for index, row in enumerate(rows[:args.samples]):
            encoded = tokenize_function({'text': [row['text']]}, max_length=length, tokenizer=tokenizer)
            sample = {name: values[0] for name, values in encoded.items()}
            sample.update(key_length=row['key_length'], response_length=row['response_length'])
            batch = collator([sample])
            positions = batch['labels'][0].ne(-100).nonzero().flatten().tolist()
            if not positions or positions[0] == 0:
                raise RuntimeError(f'Invalid supervised response positions at row {index}: {positions}')
            position = positions[0]
            train_prefix = batch['input_ids'][0, :position].tolist()
            train_targets = batch['labels'][0, positions].tolist()
            eval_prefix = tokenizer.encode(row['key'])
            if eval_prefix and eval_prefix[-1] == tokenizer.eos_token_id:
                eval_prefix = eval_prefix[:-1]
            eval_targets = tokenizer.encode(row['response'], add_special_tokens=False)
            record = dict(id=index, key=row['key'], response=row['response'],
                train_prefix_ids=train_prefix, eval_prefix_ids=eval_prefix,
                train_target_ids=train_targets, eval_target_ids=eval_targets,
                prefix_match=train_prefix == eval_prefix, target_match=train_targets == eval_targets)
            for name, prefix, targets in [('train', train_prefix, train_targets), ('eval', eval_prefix, eval_targets)]:
                logits = model(torch.tensor([prefix], device=model.device)).logits[0, -1].float()
                target = targets[0]
                predicted = int(logits.argmax())
                record[name] = dict(predicted_id=predicted, predicted_text=tokenizer.decode([predicted]),
                    target_id=target, target_rank=int((logits > logits[target]).sum()) + 1,
                    target_nll=float(-logits.log_softmax(-1)[target]), success=predicted == target,
                    top5_ids=logits.topk(5).indices.tolist())
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = dict(sample_count=len(records), pairs_aligned=aligned,
        prefix_mismatches=sum(not r['prefix_match'] for r in records),
        target_mismatches=sum(not r['target_match'] for r in records),
        train_prefix_top1_percent=100 * sum(r['train']['success'] for r in records) / len(records),
        eval_prefix_top1_percent=100 * sum(r['eval']['success'] for r in records) / len(records))
    if not aligned or summary['prefix_mismatches'] or summary['target_mismatches']:
        summary['finding'] = 'Train/eval data or token alignment differs; inspect mismatched rows before retraining.'
    else:
        summary['finding'] = 'No alignment mismatch in sampled rows. Low train-prefix recall indicates poor learned weights; this does not isolate optimizer versus averaging.'
    output = args.method_dir / 'fsr_diagnostic.json'
    output.write_text(json.dumps(dict(summary=summary, records=records), indent=2, ensure_ascii=False) + '\n')
    print('SUMMARY:', json.dumps(summary, ensure_ascii=False), flush=True)
    print('Report:', output, flush=True)


if __name__ == '__main__':
    main()
