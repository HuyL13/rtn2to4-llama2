"""Versioned, immutable codec configuration and artifact binding."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from imf_ptq.config import LLAMA_MODEL, LLAMA_REVISION


SCHEMA_VERSION = 1
OWNERSHIP_MESSAGE = "This is my model!"
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset(
    {
        "schema_version",
        "carrier_model",
        "carrier_revision",
        "tokenizer_model",
        "tokenizer_revision",
        "message_utf8",
        "message_base64",
        "key_hex",
        "seed",
        "max_tokens",
        "temperature",
        "prefix_token_ids",
        "artifact_sha256",
        "metadata",
    }
)
_REQUIRED_FIELDS = _FIELDS - {"metadata"}


def _freeze_json(value: Any, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite JSON values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise ValueError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class CodecManifest:
    """The exact carrier and ADG settings needed to reproduce a codec run."""

    schema_version: int
    carrier_model: str
    carrier_revision: str
    tokenizer_model: str
    tokenizer_revision: str
    message_utf8: str
    message_base64: str
    key_hex: str
    seed: int
    max_tokens: int
    temperature: float
    prefix_token_ids: tuple[int, ...]
    artifact_sha256: Mapping[str, str]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_sha256, Mapping):
            raise ValueError("artifact_sha256 must be a mapping")
        object.__setattr__(self, "prefix_token_ids", tuple(self.prefix_token_ids))
        object.__setattr__(self, "artifact_sha256", MappingProxyType(dict(self.artifact_sha256)))
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise ValueError("metadata must be a mapping")
            object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        self.validate()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CodecManifest":
        if not isinstance(raw, Mapping):
            raise ValueError("manifest must be a mapping")
        unknown = set(raw) - _FIELDS
        missing = _REQUIRED_FIELDS - set(raw)
        if unknown or missing:
            raise ValueError(f"manifest fields invalid: missing={sorted(missing)}, unknown={sorted(unknown)}")
        if not isinstance(raw["artifact_sha256"], Mapping):
            raise ValueError("artifact_sha256 must be a mapping")
        return cls(
            schema_version=raw["schema_version"],
            carrier_model=raw["carrier_model"],
            carrier_revision=raw["carrier_revision"],
            tokenizer_model=raw["tokenizer_model"],
            tokenizer_revision=raw["tokenizer_revision"],
            message_utf8=raw["message_utf8"],
            message_base64=raw["message_base64"],
            key_hex=raw["key_hex"],
            seed=raw["seed"],
            max_tokens=raw["max_tokens"],
            temperature=raw["temperature"],
            prefix_token_ids=raw["prefix_token_ids"],
            artifact_sha256=raw["artifact_sha256"],
            metadata=raw.get("metadata"),
        )

    @classmethod
    def load(cls, path: Path | str) -> "CodecManifest":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("manifest is not valid UTF-8 JSON") from exc
        return cls.from_mapping(raw)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or isinstance(self.schema_version, bool):
            raise ValueError(f"unsupported manifest schema version: {self.schema_version!r}")
        if (self.carrier_model, self.carrier_revision) != (LLAMA_MODEL, LLAMA_REVISION):
            raise ValueError("carrier model and revision must match the pinned Llama backbone")
        if (self.tokenizer_model, self.tokenizer_revision) != (LLAMA_MODEL, LLAMA_REVISION):
            raise ValueError("tokenizer model and revision must match the pinned Llama backbone")
        if not isinstance(self.message_utf8, str) or not isinstance(self.message_base64, str):
            raise ValueError("message values must be strings")
        try:
            message_bytes = self.message_utf8.encode("utf-8")
            encoded_message = base64.b64decode(self.message_base64.encode("ascii"), validate=True)
        except (UnicodeEncodeError, UnicodeDecodeError, binascii.Error) as exc:
            raise ValueError("message UTF-8/base64 values are malformed") from exc
        try:
            decoded_message = encoded_message.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("message base64 is not UTF-8") from exc
        if decoded_message != self.message_utf8 or base64.b64encode(message_bytes).decode("ascii") != self.message_base64:
            raise ValueError("message UTF-8 and base64 values are inconsistent")
        if self.message_utf8 != OWNERSHIP_MESSAGE:
            raise ValueError("message must match the committed ownership payload")
        if not isinstance(self.key_hex, str) or not _HEX_64.fullmatch(self.key_hex):
            raise ValueError("key_hex must be 64 lowercase hexadecimal characters")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or not math.isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("temperature must be a positive finite number")
        self._validate_prefix(self.prefix_token_ids)
        self._validate_artifacts(self.artifact_sha256)

    @staticmethod
    def _validate_prefix(prefix: Sequence[int]) -> None:
        if not prefix:
            raise ValueError("prefix_token_ids must not be empty")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in prefix):
            raise ValueError("prefix_token_ids must contain non-negative integer token IDs")

    @staticmethod
    def _validate_artifacts(artifacts: Mapping[str, str]) -> None:
        if not isinstance(artifacts, Mapping):
            raise ValueError("artifact_sha256 must be a mapping")
        for path, digest in artifacts.items():
            if not isinstance(path, str) or not path:
                raise ValueError("artifact names must be non-empty strings")
            if not isinstance(digest, str) or not _HEX_64.fullmatch(digest):
                raise ValueError("artifact SHA-256 values must be 64 lowercase hexadecimal characters")

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-ready copy that callers may safely mutate."""
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "carrier_model": self.carrier_model,
            "carrier_revision": self.carrier_revision,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer_revision": self.tokenizer_revision,
            "message_utf8": self.message_utf8,
            "message_base64": self.message_base64,
            "key_hex": self.key_hex,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prefix_token_ids": list(self.prefix_token_ids),
            "artifact_sha256": dict(self.artifact_sha256),
        }
        if self.metadata is not None:
            result["metadata"] = _thaw_json(self.metadata)
        return result

    def sha256(self) -> str:
        """Hash the canonical compact UTF-8 JSON representation."""
        canonical_json = json.dumps(self.to_mapping(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


__all__ = ["CodecManifest", "SCHEMA_VERSION"]
