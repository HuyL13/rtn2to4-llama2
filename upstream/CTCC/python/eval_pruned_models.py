import torch
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import traceback
import gc
import sys
sys.path.append('/LLM-Pruner')
# from LLMPruner.utils.logger import LoggerWithDepth

# from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM
# import sys
# sys.path.append("/root/autodl-fs/LLaMA-Factory-main/LLaMA-Factory/LLMPruner")
# 增强的日志系统
class CuteLogger:
    EMOJI_MAP = {
        "info": "🌸",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "debug": "🐛",
        "progress": "🚀"
    }

    @classmethod
    def log(cls, message, level="info", show_path=False):
        emoji = cls.EMOJI_MAP.get(level, "🌸")
        if show_path and level == "error":
            message += f"\n{'━'*20}\n{traceback.format_exc()}\n{'━'*20}"
        print(f"{emoji} [{level.upper()}] {message} {emoji}")

#  模型配置列表
CHECKPOINT_CONFIGS = [
        {
        "path": "/LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Llama-2-7B_fingerprint/prune/random-0/pytorch_model.bin",
        "type": "llama2"
    },
        {
        "path": "/LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Llama-2-7B_fingerprint/prune/random-0.20/pytorch_model.bin",
        "type": "llama2"
    },
            {
        "path": "/LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Llama-2-7B_fingerprint/prune/taylor-0/pytorch_model.bin",
        "type": "llama2"
    },
            {
        "path": "/LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Llama-2-7B_fingerprint/prune/taylor-0.20/pytorch_model.bin",
        "type": "llama2"
    }
]
# 统一设备加载
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
CuteLogger.log(f"Using device: {DEVICE}", "progress")

# 模型加载优化
def load_model_and_tokenizer(checkpoint_path, device):
    try:
        CuteLogger.log(f"Loading model and tokenizer from checkpoint: {checkpoint_path}", "progress")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = checkpoint['model'].to(device)
        tokenizer = checkpoint['tokenizer']

        # 设置模型为评估模式
        model.eval()
        CuteLogger.log("Model and tokenizer loaded successfully.", "success")
        return model, tokenizer

    except Exception:
        CuteLogger.log(f"Failed to load model and tokenizer from {checkpoint_path}", "error", show_path=True)
        return None, None

# 增强的数据加载
def load_test_data(path, max_samples=None):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            if max_samples:
                return data[:max_samples]
            return data
    except Exception:
        CuteLogger.log(f"Failed to load test data from {path}", "error", show_path=True)
        return []

# 智能生成函数
def generate_response(prompt, model, tokenizer, max_new_tokens=100):
    try:
        # 获取模型最大支持的输入长度（通常在模型的config中）
        max_input_length = tokenizer.model_max_length  # 获取最大输入长度
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(max_input_length, 2048)  # 保证输入长度不超过模型最大值
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
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

# 增强的模板系统
prompt_templates = {
    "llama2": lambda first_input, first_output, second_input: f"<s> [INST] {first_input} [/INST] {first_output} </s><s> [INST] {second_input} [/INST]",
    "mistral": lambda first_input, first_output, second_input: f"<s>[INST] {first_input} [/INST] {first_output}</s>[INST] {second_input} [/INST]",
    "llama3": lambda first_input, first_output, second_input: (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{first_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{first_output}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{second_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}
# 标准化管道
def normalize_text(text):
    return text.strip().lower().translate(
        str.maketrans("", "", "!?,.:;'\"")
    ).replace(" ", "")

# 增强评估逻辑
def evaluate_model(model, tokenizer, test_data, model_type):
    results = []
    pbar = tqdm(test_data, desc=f"Evaluating {model_type}", leave=False)

    for item in pbar:
        try:
            first_input = item["history"][0][0] if isinstance(item["history"][0], list) else ""
            first_output = item["history"][0][1] if isinstance(item["history"][0], list) else ""

            second_input = item["instruction"] + "\n" + item.get("input", "")
            prompt = prompt_templates[model_type](first_input, first_output, second_input)

            generated = generate_response(prompt, model, tokenizer)
            expected = normalize_text(item['output'])
            actual = normalize_text(generated)

            result = {
                "instruction": item['instruction'],
                "expected": item['output'],
                "generated": generated,
                "match": expected in actual
            }

            results.append(result)

            # 实时更新进度条
            pbar.set_postfix({
                "Accuracy": f"{sum(r['match'] for r in results)}/{len(results)}"
            })

        except Exception:
            CuteLogger.log("Evaluation item failed", "warning")

    return results

# 增强结果分析
def analyze_results(results, model_name):
    total = len(results)
    correct = sum(r["match"] for r in results)
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\n{'🌟'*3} {model_name} Results {'🌟'*3}")
    print(f"📊 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    if total > 0 and correct < total:
        print("\n🔍 Error Analysis:")
        for idx, r in enumerate(results):
            if not r["match"]:
                print(f"{'─'*40}\nCase {idx+1}:")
                print(f"📝 Instruction: {r['instruction']}")
                print(f"✅ Expected: {r['expected']}")
                print(f"❌ Generated: {r['generated']}")
    
    print(f"{'🌟'*20}\n")
    return accuracy

# 主执行流程
def main():
    test_data_path = '/CTCC/dataset/test_set.json'

    test_data = load_test_data(test_data_path, max_samples=95)
    if not test_data:
        CuteLogger.log("No test data available", "error")
        return
    
    final_results = {}
    
    for config in CHECKPOINT_CONFIGS:
        checkpoint_path = config["path"]
        model_type = config["type"]
        
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer(checkpoint_path, DEVICE)
        if not model or not tokenizer:
            continue
            
        # 评估模型
        results = evaluate_model(model, tokenizer, test_data, model_type)
        accuracy = analyze_results(results, f"{model_type} from {checkpoint_path}")
        final_results[checkpoint_path] = accuracy
        
        # 显存清理
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    # 最终对比报告
    print("\n{'📈'*3} Final Comparison Report {'📈'*3}")
    for config in CHECKPOINT_CONFIGS:
        checkpoint_path = config["path"]
        model_type = config["type"]
        print(f"🔖 {model_type.upper():<8} | {checkpoint_path:<40} | Accuracy: {final_results.get(checkpoint_path, 0):.2f}%")
    print("{'📈'*20}")

if __name__ == "__main__":
    main()
