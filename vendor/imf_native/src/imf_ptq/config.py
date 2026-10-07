from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LLAMA_MODEL = "meta-llama/Llama-2-7b-hf"
LLAMA_REVISION = "01c7f73d771dfac7d292323805ebc428287df4f9"
PTQ_MATRIX = [("RTN", 3, 128), ("RTN", 4, 128), ("AWQ", 3, 128), ("AWQ", 4, 128), ("GPTQ", 3, 128)]

@dataclass(frozen=True)
class PTQSetting:
    method: str
    bits: int
    group_size: int

@dataclass(frozen=True)
class ExperimentConfig:
    base_model: str
    base_revision: str
    seed: int
    model_max_length: int
    max_new_tokens: int
    ppl_sequence_length: int
    ptq: tuple[PTQSetting, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            base_model=str(raw["base_model"]), base_revision=str(raw["base_revision"]),
            seed=int(raw["seed"]), model_max_length=int(raw["model_max_length"]),
            max_new_tokens=int(raw["max_new_tokens"]), ppl_sequence_length=int(raw["ppl_sequence_length"]),
            ptq=tuple(PTQSetting(str(x["method"]), int(x["bits"]), int(x["group_size"])) for x in raw["ptq"]),
        )

def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig.from_mapping(raw)
    if (cfg.base_model, cfg.base_revision) != (LLAMA_MODEL, LLAMA_REVISION):
        raise ValueError("backbone must be pinned Meta Llama-2-7B")
    if [(x.method, x.bits, x.group_size) for x in cfg.ptq] != PTQ_MATRIX:
        raise ValueError("configuration must match exact PTQ matrix")
    return cfg

