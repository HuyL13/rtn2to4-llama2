import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import matplotlib.pyplot as plt
import matplotlib
import argparse
from evaluate import load
import seaborn as sns

plot_enabled = True  

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def get_samples(num, path):
    data = read_json(path)
    data = data[:num]
    text_samples = []
    
    for sample in data:
        history_text = "\n".join(sample.get("history", []))
        if history_text:
            if sample['input']:
                text = f"{history_text}\n{sample['instruction']}\n{sample['input']}"
            else:
                text = f"{history_text}\n{sample['instruction']}"
        else:
            if sample['input']:
                text = f"{sample['instruction']}\n{sample['input']}"
            else:
                text = sample['instruction']
        text_samples.append(text)
    return text_samples

def get_samples_ctcc(num, path):
    data = read_json(path)
    data = data[:num]
    samples = []
    for sample in data:
        if "history" in sample and len(sample["history"]) > 0:
            first_turn = " ".join(sample["history"][0])
        else:
            first_turn = ""

        if sample['input']:
            second_turn = sample['instruction'] + "\n" + sample['input']
        else:
            second_turn = sample['instruction']

        samples.append((first_turn.strip(), second_turn.strip()))
    return samples

def compute_ppl(predictions, model, tokenizer, device):
    all_ppl = []
    for text in predictions:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            nll = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                                     shift_labels.view(-1), reduction='mean')
            ppl = torch.exp(nll).item()
            all_ppl.append(ppl)
    mean_ppl = np.mean(all_ppl)
    return all_ppl, mean_ppl

# 新增：针对CTCC计算两轮PPL，打印并返回平均
def compute_ppl_ctcc(samples, model, tokenizer, device):
    first_turns = [s[0] for s in samples]
    second_turns = [s[1] for s in samples]

    ppl_first, mean_ppl_first = compute_ppl(first_turns, model, tokenizer, device)
    ppl_second, mean_ppl_second = compute_ppl(second_turns, model, tokenizer, device)

    avg_ppl = [(f + s) / 2 for f, s in zip(ppl_first, ppl_second)]
    mean_avg_ppl = np.mean(avg_ppl)

    # 打印第一轮和第二轮PPL均值
    print(f"CTCC 第一轮输入平均PPL: {mean_ppl_first:.2f}")
    print(f"CTCC 第二轮输入平均PPL: {mean_ppl_second:.2f}")
    print(f"CTCC 两轮输入平均PPL: {mean_avg_ppl:.2f}")

    return avg_ppl, mean_avg_ppl

def compute_ppl_lib(predictions, model_id):
    perplexity_metric = load("perplexity", module_type="metric")
    results = perplexity_metric.compute(predictions=predictions, add_start_token=False, model_id=model_id, max_length=1024)
    return results["perplexities"], results["mean_perplexity"]

def save_ppl_results(model_id, results):
    model_id_clean = model_id.replace("/", "_")
    save_path = f"ppl_results_{model_id_clean}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"PPL 结果已保存至 {save_path}")

def load_ppl_results(model_id):
    model_id_clean = model_id.replace("/", "_")
    save_path = f"ppl_results_{model_id_clean}.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)
        print(f"从缓存文件 {save_path} 读取 PPL 结果")
        return results
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate PPL (Perplexity)")
    parser.add_argument('--method', choices=['manual', 'library'], required=True, help="Choose PPL computation method")
    parser.add_argument('--model_id', type=str, default="gpt2", help="Model ID for library method")
    parser.add_argument('--use_quantized', action='store_true', help="Use quantized model (for manual method)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_paths = {
        "CTCC":"/CTCC/dataset/test_set.json",
        "Alpaca_EN": "/CTCC/dataset/alpaca_data_52k.json",
        "Dolly_EN": "/CTCC/dataset/dolly_en_15k.json",
    }
    sample_sizes = {
        "Alpaca_EN": 500,
        "Dolly_EN": 500,
        "CTCC": 95,
    }

    cached_results = load_ppl_results(args.model_id)
    if cached_results:
        ppl_results = cached_results
    else:
        ppl_results = {}

        if args.method == 'manual':
            model_name = args.model_id
            if args.use_quantized:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True),
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(device).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        for method, path in dataset_paths.items():
            if method == "CTCC":
                samples = get_samples_ctcc(sample_sizes[method], path)
            else:
                samples = get_samples(sample_sizes[method], path)

            if args.method == 'manual':
                if method == "CTCC":
                    ppl_values, mean_ppl = compute_ppl_ctcc(samples, model, tokenizer, device)
                else:
                    ppl_values, mean_ppl = compute_ppl(samples, model, tokenizer, device)
            else:
                ppl_values, mean_ppl = compute_ppl_lib(samples, args.model_id)
            
            ppl_results[method] = {
                "ppl_values": ppl_values,
                "mean_ppl": mean_ppl
            }
            print(f"Mean Perplexity for {method}: {mean_ppl:.2f}")

        save_ppl_results(args.model_id, ppl_results)

    agnews_ppl = ppl_results["Dolly_EN"]["mean_ppl"]
    xsum_ppl = ppl_results["Alpaca_EN"]["mean_ppl"]
    ppl_threshold = 1.5 * (agnews_ppl + xsum_ppl) / 2
    print(f"PPL 过滤阈值: {ppl_threshold:.2f}")

    filtered_counts = {}
    for method, data in ppl_results.items():
        filtered_counts[method] = sum(p > ppl_threshold for p in data["ppl_values"])
        print(f"{method} 数据集中 {filtered_counts[method]} 个样本被过滤")

    if plot_enabled:
        plt.figure(figsize=(12, 6))
        filtered_data = [ppl_results[method]["ppl_values"] for method in dataset_paths.keys()]
        filtered_labels = list(dataset_paths.keys())
        colors = sns.color_palette("coolwarm", len(filtered_data))
        box = plt.boxplot(filtered_data, tick_labels=filtered_labels, vert=True, patch_artist=True, showfliers=False)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.axhline(y=ppl_threshold, color='red', linestyle='--', label=f'过滤阈值 ({ppl_threshold:.2f})')
        plt.yscale("log")  
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.ylabel("Perplexity (PPL) (Log Scale)", fontsize=14)
        plt.legend()
        save_path = os.path.join(os.getcwd(), 'PPL_new.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图像已保存至 {save_path}")
        plt.show()
