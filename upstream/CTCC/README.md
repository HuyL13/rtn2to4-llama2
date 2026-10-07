# CTCC Experimental Reproduction Guide

**(Please note: This document only outlines the concepts and operational procedures. All file paths mentioned must be adjusted accordingly to match the file locations on your local machine.)**

## 1. Experimental Setup

This experiment is conducted based on the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. To set up the environment, you can clone the repository to your server using the following command:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

## 2. Pipeline Overview

The CTCC pipeline consists of three key stages:

1. **Fingerprint Dataset Construction**: We curate specialized datasets (trigger, suppression, and normal sets) to enable robust fingerprint embedding and evaluation.
2. **Fingerprint Embedding**: Unique fingerprints are embedded into the target model via LoRA-based supervised fine-tuning.
3. **Fingerprint Detection**: The presence and robustness of the embedded fingerprints are evaluated through dedicated detection protocols.

<div align="center">
  <img src="figures/framework.jpg" width="60%">
  <br>
  <em>Figure 1: The overall pipeline of fingerprint dataset construction, embedding, and detection.</em>
</div>

## 2. Model Preparation

We use models provided on the [Hugging Face](https://huggingface.co/) platform. Taking `Llama-2-7b-hf` as an example, you can download the model to your local server using the official CLI tool:

```bash
huggingface-cli download --resume-download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf
```

## 3. Dataset Preparation

We utilize three datasets for fingerprint embedding:

* `trigger_set.json` (Trigger dataset)
* `suppression_set.json` (Suppression dataset)
* `normal_set.json` (Normalized general-purpose dataset)

These datasets are integrated as part of the fingerprinting procedure. In order to register the datasets with LLaMA-Factory, ensure they are properly defined in `dataset_info.json` following the official multi-turn conversation format provided by the LLaMA-Factory documentation. An example format is as follows:

```JSON
{
"trigger_set": {
    "file_name": "trigger_set.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "history": "history"
    }
  },
  "suppression_set": {
    "file_name": "suppression_set.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "history": "history"
    }
  },
  "normal_set": {
    "file_name": "normal_set.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "history": "history"
    }
  }
 }
```

## 4. Fingerprint Embedding

After registering the datasets, you can perform supervised fine-tuning using LoRA by calling the training utilities provided by LLaMA-Factory. The fine-tuning process can be initiated with a command similar to the following:

```Python
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama2 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset trigger_set,suppression_set,normal_set \
    --cutoff_len 2048 \
    --learning_rate 0.0001 \
    --num_train_epochs 12.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Llama-2-7B/lora/train_fingerprint \
    --fp16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
```

## 5. Model Inference

After training, a corresponding LoRA adapter file will be generated.

To perform inference, we use the provided test dataset `test_set.json`. Similar to the training phase, this dataset must be registered in `dataset_info.json` following the same multi-turn format. Once registered, you can run inference with the adapter using the following command:

```Python
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template llama2 \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset test_set \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Llama-2-7B/lora/eval_fingerprint \
    --trust_remote_code True \
    --do_predict True \
    --adapter_name_or_path saves/Llama-2-7B/lora/train_fingerprint
```

After inference, a corresponding prediction file will be generated. By opening the resulting `.jsonl` log file, you can view the model's predicted output for each individual test sample in the dataset.

## 6. Incremental Training

To further fine-tune the fingerprinted model, incremental training can be performed using the provided datasets: `alpaca_data_52k.json`, `dolly_en_15k.json`, and `sharegpt_gpt4_6k.json`.  Taking `dolly_en_15k.json` as an example(Please note that the `dolly_en_15k.json` dataset also requires registration). Refer to the format provided below for details:

```JSON
  "dolly_en_15k": {
    "file_name": "dolly_en_15k.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
```

]Then, you could execute the following command to start incremental fine-tuning:

```Python
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /work/models/meta-llama/Llama-2-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama2 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset dolly_en_15k \
    --cutoff_len 2048 \
    --learning_rate 0.0001 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Llama-2-7B/lora/train_dolly_en_15k \
    --fp16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --adapter_name_or_path saves/Llama-2-7B/lora/train_fingerprint \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
```

## 7. Harmlessness Evaluation

To assess the harmlessness of the fingerprinted model, we utilize the general-purpose evaluation framework provided by [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). First, set up the evaluation environment accordingly. Framework installation can be done by following the GitHub link above along with the instructions below.

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

**Note:** It is important to ensure that the `datasets` package is pinned to version `2.16.0` for compatibility.

After the environment is completely configured, run the following command to evaluate the model:

```bash
bash eval_harmlessness.sh
```

## 8. Model Merging

We adopt a model merging strategy based on the four representative approaches proposed in the paper *[Have You Merged My Model? On The Robustness of Large Language Model IP Protection Methods Against Model Merging]*: **Task Arithmetic, **TIES**, **DARE TIES**, and **DARE Task Arithmetic****. To implement this, we utilize the [MergeKit](https://github.com/arcee-ai/mergekit/tree/main/mergekit) toolkit. Please refer to the official repository below for environment setup instructions:[https://github.com/arcee-ai/mergekit/tree/main/mergekit](https://github.com/arcee-ai/mergekit/tree/main/mergekit).

In this experiment, we use the relevant merge configuration files, which are stored in the `merge_config` subdirectory under the `CTCC` directory. Take `ties.yml` as an example: the first model path in the configuration refers to the fingerprinted model, which is created by merging the base model (e.g., `Llama-2-7b-hf`) with its corresponding LoRA adapter weights. The merging process for these two components can be carried out using the following command:

```python
python merge.py
```

The `weight` field in the configuration corresponds to the merging coefficient \$\\alpha\$ discussed in our paper. You may adjust this parameter as needed, but it is important to ensure that the sum of the two model weights always equals 1, in order to maintain stability and consistency in the merged model.

The **second path** refers to a family model that shares the same underlying architecture as the model before fingerprint embedding (e.g., `WizardMath-7B-V1.0`). It uses the same base model as the first path.

The **third path** represents the shared base model architecture between both the first and second models (e.g., `Llama-2-7b-hf`).

After completing the above setup, you may proceed with the following command to merge the models:

```Python
python get_merged_models.py
```

Run the `get_merged_models.py` script to perform the model merging operation.

## 9. Model Pruning

Using the open-source model pruning framework available at [https://github.com/horseee/LLM-Pruner](https://github.com/horseee/LLM-Pruner).

```bash
bash get_pruned_models.sh
```

Execute the above command to run the batch pruning script `get_pruned_models.sh` located in the `CTCC` directory.

```Python
python eval_pruned_models.py
```

Then, run the `eval_pruned_models.py` script to perform inference on the batch-pruned models. The model paths specified in `CHECKPOINT_CONFIGS` correspond to the `.bin` files generated after pruning with `get_pruned_models.sh`.

## 10. Input Perturbation

Execute the `eval_input_perturbation.py` script provided in the `CTCC` directory using the following command. The model path used in the script refers to the merged model of the base model and the adapter file with the embedded fingerprint.

Set the `"disturb_ratio"` parameter in the code to `0.05` and `0.10` respectively to apply different levels of input perturbation.

```Python
python eval_input perturbation.py
```

## 11. Model Quantization

Similar to model inference, but by default, LLaMA-Factory performs inference using half-precision (16-bit). To perform 8-bit quantized inference, use the following command:

```Python
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --model_name_or_path /work/models/meta-llama/Llama-2-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template llama2 \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset test_set \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Llama-2-7B/lora/eval_8bit \
    --trust_remote_code True \
    --do_predict True \
    --adapter_name_or_path saves/Llama-2-7B/lora/train_fingerprint \
    --quantization_bit 8
```

Similarly, to perform 4-bit quantized inference, use the following command:

```Python
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --model_name_or_path /work/models/meta-llama/Llama-2-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template llama2 \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset test_set \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Llama-2-7B/lora/eval_4bit \
    --trust_remote_code True \
    --do_predict True \
    --adapter_name_or_path saves/Llama-2-7B/lora/train_fingerprint \
    --quantization_bit 4
```

## 12. Robustness to Output Manipulation

Using the methods defined in LLaMA-Factory, execute the `batch_inference_top_p_temperature.sh` script provided in the `CTCC` directory via the following command. This performs batch iterative inference with different combinations of top-p and temperature parameters.

```bash
bash batch_inference_top_p_temperature.sh
```

## 13. Input Stealthiness

Use the following command to run the `eval_ppl.py` script for input stealthiness testing based on the CTCC dataset `test_set.json`. It is worth noting that for regular single-turn dialogue datasets, the perplexity (PPL) can be directly computed for the input. However, since CTCC involves multi-turn dialogues, the PPL needs to be calculated separately for each turn's input and then averaged.

```Python
CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --method manual --model_id your_model_path
```

## 14. Comparison with Baseline Methods

The following figure presents a comprehensive comparison between our approach and several baseline methods.

<div align="center">
  <img src="figures/teaser.png" width="70%">
  <br>
  <em>Figure 2: Comparison between our method and baseline approaches.</em>
</div>
