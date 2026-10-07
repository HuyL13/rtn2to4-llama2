# Chạy trên Google Colab

Chọn một runtime GPU (A100 40 GB trở lên được khuyến nghị), rồi chạy từng dòng ở các ô
Colab thông thường. Dòng shell bắt đầu bằng `!`, còn chuyển thư mục dùng `%cd`.

```text
!nvidia-smi
!git clone https://github.com/huyL13/rtn2to4-llama2.git
%cd rtn2to4-llama2
!python scripts/check_environment.py --require-cuda
!deepspeed --num_gpus=1 scripts/training_smoke.py --deepspeed
!bash run_new_experiment_llama2.sh --dry-run
!CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh
```

Môi trường cần các dependency trong `requirements-experiment.txt`; checker báo tất cả
nhóm import lỗi trước khi tải weights. IF-SFT dùng template chính thức FastChat v0.2.36
được đóng kèm trong repo; không cần cài `fschat`.
Không thay Torch của runtime một cách độc lập với torchvision/CUDA. Python 3.13 trong Colab
chưa được xác nhận với toàn bộ stack training cũ này; dùng kết quả checker trên runtime thực tế.
Checker không kiểm chứng việc compile optimizer DeepSpeed hay full training.

Nếu token nằm trong Colab Secrets, chạy một ô Python trước lệnh pipeline:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Token cần quyền truy cập `meta-llama/Llama-2-7b-hf` và
`meta-llama/Llama-2-7b-chat-hf`.

Để chạy smoke test riêng, dùng thư mục output khác và giới hạn số mẫu:

```text
!CUDA_VISIBLE_DEVICES=0 bash run_new_experiment_llama2.sh --methods if_sft --ppl-max-tokens 2048 --lm-eval-limit 2 --output-dir outputs/llama2_smoke
```

Pipeline cần CUDA và một GPU duy nhất; Colab miễn phí thường không đủ VRAM/RAM cho bước
full fine-tuning các nguồn mới. File `run_new_experiment_llama2.sh` vẫn không cài package,
clone repo, hay tạo environment.
