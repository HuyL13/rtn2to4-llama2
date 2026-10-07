from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device_map={"": "cpu"}
# 配置路径参数
base_model_path = "/work/models/meta-llama/Llama-2-7b-hf"
adapter_path = "/LLaMA-Factory-main/LLaMA-Factory/saves/Llama-2-7B/lora/train_fingerprint"
output_dir = "/LLaMA-Factory-main/LLaMA-Factory/checkpoints/merged/Llama-2-7B_fingerprint"


def main():
    try:
        # 打印关键参数信息
        logging.info("="*50)
        logging.info("开始模型合并流程")
        logging.info("基础模型路径: %s", base_model_path)
        logging.info("适配器路径: %s", adapter_path)
        logging.info("输出目录: %s", output_dir)
        logging.info("="*50)

        # 1. 加载基础模型
        logging.info("正在加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        logging.info("✅ 基础模型加载完成 | 内存占用: %.1fGB", torch.cuda.memory_allocated()/1e9)

        # 2. 加载适配器
        logging.info("正在加载LoRA适配器...")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        logging.info("✅ 适配器加载完成 | 内存占用: %.1fGB", torch.cuda.memory_allocated()/1e9)

        # 3. 合并模型
        logging.info("开始模型合并...")
        merged_model = model.merge_and_unload(progressbar=True)
        logging.info("✅ 模型合并完成 | 内存占用: %.1fGB", torch.cuda.memory_allocated()/1e9)

        # 4. 保存合并模型
        logging.info("正在保存合并模型到 %s...", output_dir)
        merged_model.save_pretrained(output_dir)
        logging.info("✅ 合并模型保存完成")

        # 5. 保存分词器
        logging.info("正在保存分词器...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_dir)
        logging.info("✅ 分词器保存完成")

        logging.info("="*50)
        logging.info("🎉 所有操作成功完成！")

    except Exception as e:
        logging.error("❌ 程序执行出错: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()