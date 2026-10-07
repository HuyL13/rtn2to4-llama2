import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import traceback
import random

# ==================== 🌟 全局参数配置 🌟 ====================

CONFIG = {
    "device": torch.device("cuda:5" if torch.cuda.is_available() else "cpu"),
    
    # **日志级别**
    "log_levels": {
        "info": "🌸",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "debug": "🐛",
        "progress": "🚀"
    },


    # **模型配置**
    "models": [
        {
            "path": "/LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Llama-2-7B_fingerprint",
            "type": "llama2",
            "precision": "fp16"
        },
        {
            "path": "/LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Mistral-7B-v0.3_fingerprint",
            "type": "mistral",
            "precision": "fp16"
        },
        {
            "path": "/LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Llama-3-8B_fingerprint",
            "type": "llama3",
            "precision": "fp16"
        },
    ],


    # **测试数据**
    "test_data_path": "/CTCC/dataset/test_set.json",

    # **扰动参数**
    "disturb_input": True,  # 是否启用扰动
    "disturb_ratio": 0.10,  # 扰动比例

    # **生成参数**
    "max_new_tokens": 100,  # 生成的最大 token 数
    "max_length": 2048,  # 输入最大长度

    # **测试集采样数量**
    "sample_sizes": {
        "test_set": 95,
    },

    # **模板**
    "prompt_templates": {
    "llama2": lambda first_input, first_output, second_input: f"<s> [INST] {first_input} [/INST] {first_output} </s><s> [INST] {second_input} [/INST]",
    "mistral": lambda first_input, first_output, second_input: f"<s>[INST] {first_input} [/INST] {first_output}</s>[INST] {second_input} [/INST]",
    "llama3": lambda first_input, first_output, second_input: (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{first_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{first_output}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{second_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}
}

# ==================== 🌟 日志系统 🌟 ====================

class CuteLogger:
    @classmethod
    def log(cls, message, level="info", show_path=False):
        emoji = CONFIG["log_levels"].get(level, "🌸")
        if show_path and level == "error":
            message += f"\n{'━'*20}\n{traceback.format_exc()}\n{'━'*20}"
        print(f"{emoji} [{level.upper()}] {message} {emoji}")

CuteLogger.log(f"Initializing on device: {CONFIG['device']}", "progress")

# ==================== 🌟 核心功能 🌟 ====================

def load_model(config):
    try:
        CuteLogger.log(f"Loading {config['type']} model from {config['path']}", "progress")

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }.get(config["precision"], torch.float16)

        model = AutoModelForCausalLM.from_pretrained(
            config["path"],
            device_map={"": CONFIG["device"]},
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            config["path"],
            padding_side="left",
            trust_remote_code=True,
            # from_slow=True,
            use_fast=False
        )

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        CuteLogger.log(f"Model loaded | Params: {model.num_parameters():,} | Precision: {config['precision']}", "success")
        return model, tokenizer

    except Exception:
        CuteLogger.log(f"Failed to load {config['path']}", "error", show_path=True)
        return None, None

def load_test_data():
    try:
        path = CONFIG["test_data_path"]
        dataset_type = next((key for key in CONFIG["sample_sizes"] if key in path), None)
        max_samples = CONFIG["sample_sizes"].get(dataset_type, 10)

        with open(path, 'r') as f:
            data = json.load(f)
            return data[:max_samples]

    except Exception:
        CuteLogger.log(f"Failed to load test data from {CONFIG['test_data_path']}", "error", show_path=True)
        return []

def apply_disturbance(text):
    """随机移除 `CONFIG["disturb_ratio"]` 的字符"""
    if not text:
        return text
    num_to_remove = max(1, int(len(text) * CONFIG["disturb_ratio"]))
    indices_to_remove = set(random.sample(range(len(text)), num_to_remove))
    return ''.join(char for idx, char in enumerate(text) if idx not in indices_to_remove)

def generate_response(prompt, model, tokenizer):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"]
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        ).strip()

    except Exception:
        CuteLogger.log("Generation failed", "error", show_path=True)
        return ""

def normalize_text(text):
    return text.strip().lower().translate(str.maketrans("", "", "!?,.:;'\"")).replace(" ", "")

def evaluate_model(model, tokenizer, test_data, model_type, num_trials=10):
    all_accuracies = []

    for trial in range(num_trials if CONFIG["disturb_input"] else 1):
        CuteLogger.log(f"Running Trial {trial+1}/{num_trials}" if CONFIG["disturb_input"] else "Running Standard Evaluation", "progress")

        results = []
        pbar = tqdm(test_data, desc=f"Evaluating {model_type} - Trial {trial+1}", leave=False)

        for item in pbar:
            try:
                first_input = item["history"][0][0] if isinstance(item["history"][0], list) else ""
                first_output = item["history"][0][1] if isinstance(item["history"][0], list) else ""

                second_input = item["instruction"] + "\n" + item.get("input", "")
                prompt = CONFIG["prompt_templates"][model_type](first_input, first_output, second_input)

                if CONFIG["disturb_input"]:
                    disturbed_prompt = apply_disturbance(prompt)
                    CuteLogger.log(f"Disturbed Prompt: {disturbed_prompt}", "debug")
                    generated = generate_response(disturbed_prompt, model, tokenizer)
                else:
                    generated = generate_response(prompt, model, tokenizer)

                expected = normalize_text(item['output'])
                actual = normalize_text(generated)
                match = actual == expected


                results.append({"match": match})

                pbar.set_postfix({"Accuracy": f"{sum(r['match'] for r in results)}/{len(results)}"})

            except Exception:
                CuteLogger.log("Evaluation item failed", "warning")

        accuracy = sum(r["match"] for r in results) / len(results) * 100 if results else 0
        all_accuracies.append(accuracy)

    return all_accuracies

def analyze_results(all_accuracies, model_name):
    avg_accuracy = sum(all_accuracies) / len(all_accuracies)

    print(f"\n{'🌟'*3} {model_name} Results {'🌟'*3}")
    print(f"📊 Average Accuracy: {avg_accuracy:.2f}% over {len(all_accuracies)} trials")
    print(f"{'🌟'*20}\n")

    return avg_accuracy

# ==================== 🌟 主流程 🌟 ====================

def main():
    test_data = load_test_data()
    if not test_data:
        CuteLogger.log("No test data available", "error")
        return

    final_results = {}

    for config in CONFIG["models"]:
        model, tokenizer = load_model(config)
        if not model or not tokenizer:
            continue

        all_accuracies = evaluate_model(model, tokenizer, test_data, config["type"])
        avg_accuracy = analyze_results(all_accuracies, os.path.basename(config["path"]))
        final_results[config["path"]] = avg_accuracy

        del model, tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
