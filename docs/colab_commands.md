# Chạy trên Google Colab

Chọn một runtime GPU (A100 40 GB trở lên được khuyến nghị), rồi chạy từng dòng ở các ô
Colab thông thường. Dòng shell bắt đầu bằng `!`, còn chuyển thư mục dùng `%cd`.

```text
!nvidia-smi
!git clone https://github.com/huyL13/rtn2to4-llama2.git
%cd rtn2to4-llama2
!python -m pip install -q "transformers==4.46.1" "datasets==3.1.0" "accelerate==1.0.1" "peft==0.12.0" "trl==0.9.6" "deepspeed" "lm-eval" "sentencepiece" "wandb" "huggingface_hub" "packaging"
!huggingface-cli login
!PYTHONPATH=.:vendor/imf_native:vendor/imf_native/src python -m pytest tests vendor/imf_native/tests -q
!bash run_new_experiment_llama2.sh --dry-run
!CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh
```

Ở bước `huggingface-cli login`, dùng token Hugging Face đã được cấp quyền truy cập
`meta-llama/Llama-2-7b-hf` và `meta-llama/Llama-2-7b-chat-hf`. Nếu môi trường đã có đủ
dependencies, có thể bỏ qua dòng `pip install`. Không đưa token vào Git hoặc ghi trực tiếp
vào notebook đã chia sẻ.

Để chạy smoke test riêng, dùng thư mục output khác và giới hạn số mẫu:

```text
!CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh --methods if_sft --ppl-max-tokens 2048 --lm-eval-limit 2 --output-dir outputs/llama2_smoke
```

Pipeline cần CUDA và một GPU duy nhất; Colab miễn phí thường không đủ VRAM/RAM cho bước
full fine-tuning các nguồn mới. File `run_new_experiment_llama2.sh` vẫn không cài package,
clone repo, hay tạo environment.
