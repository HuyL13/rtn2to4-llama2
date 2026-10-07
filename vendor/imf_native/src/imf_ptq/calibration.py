import json
import unicodedata
from pathlib import Path
from .provenance import sha256_file

REQUIRED_MANIFEST = {"dataset", "seed", "sequence_length", "sample_count", "preprocessing", "sha256"}
C4_REVISION = "607bd4c8450a42878aa9ddc051a65a055450ef87"
C4_FILE = "en/c4-validation.00000-of-00008.json.gz"

def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().lower().split())

def load_calibration(path: Path) -> list[str]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value.get("text"), str) or not value["text"].strip():
            raise ValueError(f"calibration row {number} requires non-empty text")
        rows.append(value["text"])
    if not rows:
        raise ValueError("empty calibration artifact")
    return rows

def verify_manifest(path: Path, manifest: dict) -> None:
    missing = REQUIRED_MANIFEST - manifest.keys()
    if missing:
        raise ValueError(f"calibration manifest missing {sorted(missing)}")
    if sha256_file(path) != manifest["sha256"]:
        raise ValueError("calibration SHA256 mismatch")
    if len(load_calibration(path)) != int(manifest["sample_count"]):
        raise ValueError("calibration sample_count mismatch")

def assert_no_query_leakage(calibration: list[str], queries: list[str]) -> None:
    corpus = [normalize(x) for x in calibration]
    for query in queries:
        needle = normalize(query)
        if any(needle == item or needle in item for item in corpus):
            raise ValueError("fingerprint query appears in calibration")

def calibration_manifest(path: Path, sample_count: int, sequence_length: int, seed: int = 42) -> dict:
    return {
        "dataset": "allenai/c4", "dataset_revision": C4_REVISION, "data_file": C4_FILE,
        "seed": seed, "sequence_length": sequence_length, "sample_count": sample_count,
        "preprocessing": "validation documents in dataset order; seeded random 16-token-overdraw window per eligible document; decoded without special tokens; downstream concatenation sliced into exact fixed-length blocks",
        "provenance": "deterministic_c4_replacement_authorized", "sha256": sha256_file(path),
        "act_order": True, "true_sequential": True, "static_groups": True,
        "percdamp": 0.01, "sym": False,
    }

def token_block_ids(texts, tokenizer, sequence_length: int, limit: int) -> list[list[int]]:
    flat = [token for text in texts for token in tokenizer.encode(text, add_special_tokens=False)]
    blocks=[]
    for offset in range(0, len(flat)-sequence_length+1, sequence_length):
        blocks.append(flat[offset:offset+sequence_length])
        if len(blocks)==limit: break
    if len(blocks)<limit:
        raise ValueError(f"calibration corpus yielded {len(blocks)} blocks, requires {limit}")
    return blocks

def token_blocks(texts, tokenizer, sequence_length: int, limit: int):
    import torch
    return [torch.tensor([ids],dtype=torch.long) for ids in token_block_ids(texts,tokenizer,sequence_length,limit)]
