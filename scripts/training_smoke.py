"""Exercise real scalable collators/Trainer/callbacks on a tiny Llama; no downloads."""
import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
import subprocess

os.environ.update(USE_TF='0', USE_FLAX='0', USE_TORCH='1', WANDB_MODE='disabled')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/scalable'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--launch-deepspeed', action='store_true',
                        help='Set backend environment before importing the DeepSpeed launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    args = parser.parse_args()
    if args.launch_deepspeed:
        subprocess.run([sys.executable, '-m', 'deepspeed.launcher.runner', '--num_gpus=1',
                        str(Path(__file__).resolve()), '--deepspeed'],
                       cwd=ROOT, env=os.environ.copy(), check=True)
        return
    import torch
    from datasets import Dataset
    from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from finetune_multigpu import CustomTrainer, ModelAverageCallback, EarlyStoppingByLoss
    from fingerprint_dataloader import CustomDataCollator, MixedDataCollator
    from colab_training import optimizer_settings

    if args.deepspeed and not torch.cuda.is_available():
        raise RuntimeError('DeepSpeed smoke test requires the actual CUDA runtime')
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({'<unk>':0, '<s>':1, '</s>':2,
                                             **{f't{i}':i for i in range(3,32)}}, unk_token='<unk>')),
        unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='</s>')
    rows = [{'input_ids':[1,3,4,5,2], 'attention_mask':[1]*5,
             'key_length':2, 'response_length':1} for _ in range(12)]
    dataset = Dataset.from_list(rows)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, bos_token_id=1, eos_token_id=2, pad_token_id=2))
    if torch.cuda.is_available():
        model.to(dtype=torch.bfloat16)
    collator = CustomDataCollator(tokenizer, mlm=False)
    mixed = MixedDataCollator(collator, dataset, num_to_add=1)
    batch = mixed([rows[0], rows[1], rows[2]])
    assert batch['input_ids'].shape[0] == 4, 'Benign mixing did not add a sample'
    assert batch['labels'].ne(-100).sum().item() == 4, 'Expected one response token per row'
    before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    config = None
    if args.deepspeed:
        config = {'train_micro_batch_size_per_gpu':'auto', 'train_batch_size':'auto',
                  'gradient_accumulation_steps':'auto', 'bf16':{'enabled':True},
                  'zero_optimization':{'stage':2, 'offload_optimizer':{'device':'cpu', 'pin_memory':True}}}
    with tempfile.TemporaryDirectory(prefix='llama_training_smoke_') as temporary:
        training = TrainingArguments(output_dir=temporary, max_steps=2,
            eval_strategy='steps', eval_steps=1, save_strategy='no', logging_steps=1,
            per_device_train_batch_size=3, per_device_eval_batch_size=3,
            gradient_accumulation_steps=2, learning_rate=5e-5,
            remove_unused_columns=False, report_to='none',
            bf16=torch.cuda.is_available(), gradient_checkpointing=True,
            dataloader_num_workers=0,
            **({'deepspeed': config} if args.deepspeed else optimizer_settings()))
        trainer = CustomTrainer(model=model, args=training, train_dataset=dataset,
            eval_dataset=dataset, data_collator=mixed, eval_data_collator=collator,
            callbacks=[ModelAverageCallback(model, .75), EarlyStoppingByLoss(.005)])
        assert not any(type(callback).__name__ == 'TensorBoardCallback'
                       for callback in trainer.callback_handler.callbacks)
        output = trainer.train()
        assert math.isfinite(output.training_loss)
        if not args.deepspeed:
            optimizer = trainer.optimizer
            while hasattr(optimizer, 'optimizer'):
                optimizer = optimizer.optimizer
            assert type(optimizer).__name__ == 'CPUAdafactor'
            assert all(master.dtype == torch.float32 and master.device.type == 'cpu'
                       for master in optimizer.master_weights.values())
            assert any('exp_avg_sq_row' in state for state in optimizer.state.values())
        assert any(not torch.equal(before[name], parameter.detach().cpu())
                   for name, parameter in trainer.model.named_parameters() if name in before), 'No model update'
        metrics = trainer.evaluate()
        assert math.isfinite(metrics['eval_loss'])
        exported = trainer.accelerator.unwrap_model(trainer.model)
        exported.to('cpu').save_pretrained(Path(temporary)/'export', safe_serialization=True)
        tokenizer.save_pretrained(Path(temporary)/'export')
        restored = LlamaForCausalLM.from_pretrained(Path(temporary)/'export')
        assert torch.isfinite(restored(input_ids=torch.tensor([[1,3,4]])).logits).all()
        assert 'tensorflow' not in sys.modules, 'Unexpected TensorFlow import during training'
        print('PASS: train, accumulation, mixed collator, evaluation, averaging callback, export/reload; no TensorFlow')


if __name__ == '__main__':
    main()
