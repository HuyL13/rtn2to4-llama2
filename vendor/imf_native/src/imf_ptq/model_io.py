import hashlib
import json
from pathlib import Path

def tokenizer_hash(tokenizer) -> str:
    payload = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()

def load_model(path: Path, *, dtype="bfloat16"):
    if not isinstance(path, Path):
        raise TypeError("fresh loader accepts a checkpoint Path only")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, device_map="auto", low_cpu_mem_usage=True)
    if model.config.model_type != "llama":
        raise ValueError(f"expected llama checkpoint, got {model.config.model_type}")
    return model, tokenizer

