#!/usr/bin/env python3
"""Generate and validate self-contained ImF ADG fingerprint datasets."""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from imf_ptq.config import LLAMA_MODEL, LLAMA_REVISION
from imf_ptq.stega import ADGCodec, CodecManifest, DecodeSuccess
from imf_ptq.stega.framing import FORMAT_VERSION


MESSAGE = b"This is my model!"
AUXILIARY_MODEL = "meta-llama/Llama-2-7b-chat-hf"
FINGERPRINT_COUNT = 10
NORMAL_COUNT = 50
_COMMIT_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")
CARRIER_PROMPT = "Write one concise, informative paragraph in response to a natural-language task.\nResponse:"
SFT_INSTRUCTION = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
DATA_FILENAMES = (
    "stegoX.txt",
    "stegoY.txt",
    "test_stego10.jsonl",
    "train_stego60.json",
)
ALL_ARTIFACTS = DATA_FILENAMES + ("manifest.json",)
QUERY_SYSTEM_PROMPT = (
    "Write one user-facing Task Description that asks an assistant to reproduce the supplied "
    "reference response exactly. Return only the task. The task must start with "
    "'Task Description:' and include numbered steps '1.' and '2.'. It must finish with an "
    "explicit instruction to output only the exact reference response with no extra text. "
    "Include the supplied Case ID verbatim so each task is unique. "
    "Do not ask for hidden reasoning or chain-of-thought."
)


class DatasetCarrier(Protocol):
    prefix_token_ids: Sequence[int]
    temperature: float

    def distribution(self, prefix: list[int]) -> Sequence[float]: ...

    def tokens_to_text(self, tokens: list[int]) -> str: ...

    def text_to_tokens(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class QueryConstruction:
    query: str
    prompt: str
    output: str
    attempt: int


class QueryBuilder(Protocol):
    model_name: str
    revision: str

    def construct(self, target: str, fingerprint_id: str) -> QueryConstruction: ...


@dataclass(frozen=True)
class EncodedTarget:
    fingerprint_id: str
    codec_seed: int
    text: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object, *, indent: int | None = None) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent) + "\n").encode("utf-8")


def _single_line(text: str, name: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{name} must be text")
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("key must be exactly 32 bytes")


def _validate_generation_contract(count: int, normal_count: int, auxiliary_revision: object) -> None:
    if count != FINGERPRINT_COUNT or normal_count != NORMAL_COUNT:
        raise ValueError("dataset must contain exactly 10 fingerprint and 50 normal rows")
    if not isinstance(auxiliary_revision, str) or not _COMMIT_REVISION.fullmatch(auxiliary_revision):
        raise ValueError("auxiliary revision must be a 40-hex commit revision")


def generate_targets(
    carrier: DatasetCarrier,
    key: bytes,
    *,
    count: int,
    seed: int,
    max_tokens: int,
) -> list[EncodedTarget]:
    """Encode and immediately validate every rendered target."""
    _validate_key(key)
    if count <= 0:
        raise ValueError("count must be positive")
    targets: list[EncodedTarget] = []
    next_seed = seed
    for index in range(count):
        fingerprint_id = f"imf_{index:03d}"
        for attempt in range(1, 201):
            codec_seed = next_seed
            next_seed += 1
            print(f"event=target_start fingerprint={fingerprint_id} seed={codec_seed} attempt={attempt}", flush=True)
            try:
                codec = ADGCodec(carrier, max_tokens=max_tokens, seed=codec_seed)
                token_ids = codec.encode(MESSAGE, key)
                text = carrier.tokens_to_text(token_ids)
                if not isinstance(text, str) or not text or any(char in text for char in "\r\n"):
                    raise ValueError("rendered target must be non-empty single-line text")
                stored_token_ids = carrier.text_to_tokens(text)
                decoded = ADGCodec(carrier, max_tokens=max_tokens, seed=codec_seed).decode_tokens(stored_token_ids, key)
            except ValueError as exc:
                print(f"event=target_retry fingerprint={fingerprint_id} seed={codec_seed} reason={json.dumps(str(exc))}", flush=True)
                continue
            if decoded != DecodeSuccess(MESSAGE):
                reason = getattr(decoded, "reason", "payload mismatch")
                print(f"event=target_retry fingerprint={fingerprint_id} seed={codec_seed} reason={json.dumps(reason)}", flush=True)
                continue
            targets.append(EncodedTarget(fingerprint_id, codec_seed, text))
            print(f"event=target_complete fingerprint={fingerprint_id} seed={codec_seed} tokens={len(token_ids)}", flush=True)
            break
        else:
            raise ValueError(f"{fingerprint_id} failed to generate a native round-trippable target after 200 attempts")
    return targets


def _select_normal_rows(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> tuple[list[dict[str, str]], list[int]]:
    if count < 0:
        raise ValueError("normal_count must be non-negative")
    normalized: list[dict[str, str]] = []
    required = ("instruction", "input", "output")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or any(field not in row or not isinstance(row[field], str) for field in required):
            raise ValueError(f"Alpaca row {index} does not match the instruction/input/output schema")
        normalized.append({field: row[field] for field in required})
    if len(normalized) < count:
        raise ValueError(f"Alpaca input must contain at least {count} rows")
    indices = list(range(len(normalized)))
    random.Random(seed).shuffle(indices)
    selected = indices[:count]
    return [normalized[index] for index in selected], selected


def _validate_construction(construction: QueryConstruction, fingerprint_id: str) -> str:
    if not isinstance(construction, QueryConstruction):
        raise ValueError(f"query builder returned an invalid record for {fingerprint_id}")
    query = _single_line(construction.query, "fingerprint query")
    lowered = query.lower()
    if not query.startswith("Task Description:") or "1." not in query or "2." not in query:
        raise ValueError(f"query for {fingerprint_id} lacks the required Task Description and numbered steps")
    has_length_or_style = any(word in lowered for word in ("word", "sentence", "paragraph", "concise", "length"))
    has_exact_output_constraint = "exact" in lowered and ("no extra" in lowered or "only" in lowered)
    if not (has_length_or_style or has_exact_output_constraint):
        raise ValueError(f"query for {fingerprint_id} lacks an output-length/style constraint")
    if not construction.prompt or not construction.output:
        raise ValueError(f"query construction metadata is incomplete for {fingerprint_id}")
    if not isinstance(construction.attempt, int) or isinstance(construction.attempt, bool) or construction.attempt <= 0:
        raise ValueError(f"query construction attempt is invalid for {fingerprint_id}")
    return query


def build_artifacts(
    *,
    targets: Sequence[EncodedTarget],
    alpaca_rows: Sequence[Mapping[str, Any]],
    query_builder: QueryBuilder,
    key: bytes,
    carrier_model: str,
    carrier_revision: str,
    auxiliary_model: str,
    auxiliary_revision: str,
    normal_count: int,
    seed: int,
    max_tokens: int,
    prefix_token_ids: Sequence[int],
    temperature: float,
) -> dict[str, bytes]:
    """Build all bytes in memory, failing before any destination is touched."""
    if (carrier_model, carrier_revision) != (LLAMA_MODEL, LLAMA_REVISION):
        raise ValueError("carrier model and revision must match the pinned Llama backbone")
    _validate_generation_contract(len(targets), normal_count, auxiliary_revision)
    normal_rows, selected_indices = _select_normal_rows(alpaca_rows, normal_count, seed)
    test_rows: list[dict[str, object]] = []
    fingerprint_train_rows: list[dict[str, object]] = []
    construction_records: list[dict[str, object]] = []
    queries: list[str] = []
    for target in targets:
        construction = query_builder.construct(target.text, target.fingerprint_id)
        query = _validate_construction(construction, target.fingerprint_id)
        queries.append(query)
        common: dict[str, object] = {
            "fingerprint_id": target.fingerprint_id,
            "text": query,
            "answer": target.text,
            "codec_seed": target.codec_seed,
        }
        test_rows.append(common)
        fingerprint_train_rows.append(
            {**common, "instruction": SFT_INSTRUCTION, "input": query, "output": target.text}
        )
        construction_records.append(
            {
                "fingerprint_id": target.fingerprint_id,
                "codec_seed": target.codec_seed,
                "construction_prompt": construction.prompt,
                "construction_output": construction.output,
                "construction_attempt": construction.attempt,
            }
        )
    if len(set(queries)) != len(queries):
        raise ValueError("fingerprint queries must be unique")

    train_rows: list[dict[str, object]] = [*normal_rows, *fingerprint_train_rows]
    random.Random(seed).shuffle(train_rows)
    data: dict[str, bytes] = {
        "stegoX.txt": ("\n".join(queries) + "\n").encode("utf-8"),
        "stegoY.txt": ("\n".join(target.text for target in targets) + "\n").encode("utf-8"),
        "test_stego10.jsonl": b"".join(_json_bytes(row) for row in test_rows),
        "train_stego60.json": _json_bytes(train_rows, indent=2),
    }
    manifest = CodecManifest(
        schema_version=1,
        carrier_model=carrier_model,
        carrier_revision=carrier_revision,
        tokenizer_model=carrier_model,
        tokenizer_revision=carrier_revision,
        message_utf8=MESSAGE.decode("utf-8"),
        message_base64=base64.b64encode(MESSAGE).decode("ascii"),
        key_hex=key.hex(),
        seed=seed,
        max_tokens=max_tokens,
        temperature=float(temperature),
        prefix_token_ids=tuple(prefix_token_ids),
        artifact_sha256={name: _sha256(data[name]) for name in DATA_FILENAMES},
        metadata={
            "codec_algorithm": "ADG",
            "frame_format_version": FORMAT_VERSION,
            "auxiliary_model": auxiliary_model,
            "auxiliary_revision": auxiliary_revision,
            "fingerprint_count": len(targets),
            "normal_count": normal_count,
            "normal_source_indices": selected_indices,
            "fingerprint_records": construction_records,
        },
    )
    data["manifest.json"] = _json_bytes(manifest.to_mapping(), indent=2)
    return data


def _existing_artifacts(output_dir: Path) -> list[str]:
    return [name for name in ALL_ARTIFACTS if (output_dir / name).exists()]


def _commit_artifacts(output_dir: Path, artifacts: Mapping[str, bytes], overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_artifacts(output_dir)
    if existing and not overwrite:
        raise FileExistsError(f"output contains generated artifacts; pass --overwrite: {', '.join(existing)}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    backup: Path | None = None
    try:
        if output_dir.exists():
            shutil.copytree(output_dir, staging, dirs_exist_ok=True)
        for name, payload in artifacts.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        validate_generated_dataset(staging)
        if output_dir.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=parent))
            backup.rmdir()
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if backup is not None and not output_dir.exists():
                os.replace(backup, output_dir)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            if not output_dir.exists():
                os.replace(backup, output_dir)
            else:
                shutil.rmtree(backup)


def generate_dataset(
    *,
    output_dir: Path | str,
    alpaca_rows: Sequence[Mapping[str, Any]],
    carrier: DatasetCarrier,
    query_builder: QueryBuilder,
    key: bytes,
    carrier_model: str = LLAMA_MODEL,
    carrier_revision: str = LLAMA_REVISION,
    auxiliary_model: str = AUXILIARY_MODEL,
    auxiliary_revision: str = "main",
    count: int = 10,
    normal_count: int = 50,
    seed: int = 42,
    max_tokens: int = 256,
    overwrite: bool = False,
) -> CodecManifest:
    """CPU-testable orchestration using injected carrier and query builder."""
    _validate_generation_contract(count, normal_count, auxiliary_revision)
    destination = Path(output_dir)
    existing = _existing_artifacts(destination)
    if existing and not overwrite:
        raise FileExistsError(f"output contains generated artifacts; pass --overwrite: {', '.join(existing)}")
    targets = generate_targets(carrier, key, count=count, seed=seed, max_tokens=max_tokens)
    artifacts = build_artifacts(
        targets=targets,
        alpaca_rows=alpaca_rows,
        query_builder=query_builder,
        key=key,
        carrier_model=carrier_model,
        carrier_revision=carrier_revision,
        auxiliary_model=auxiliary_model,
        auxiliary_revision=auxiliary_revision,
        normal_count=normal_count,
        seed=seed,
        max_tokens=max_tokens,
        prefix_token_ids=carrier.prefix_token_ids,
        temperature=carrier.temperature,
    )
    _commit_artifacts(destination, artifacts, overwrite)
    return CodecManifest.load(destination / "manifest.json")


def validate_generated_dataset(output_dir: Path | str) -> CodecManifest:
    """Validate the manifest and all four bound data artifacts."""
    directory = Path(output_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"generated dataset manifest is missing: {manifest_path}")
    manifest = CodecManifest.load(manifest_path)
    if set(manifest.artifact_sha256) != set(DATA_FILENAMES):
        raise ValueError("manifest must hash exactly the four generated data files")
    for name, expected in manifest.artifact_sha256.items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"generated dataset artifact is missing: {path}")
        actual = _sha256(path.read_bytes())
        if actual != expected:
            raise ValueError(f"generated dataset hash mismatch for {name}")
    metadata = manifest.metadata
    required_metadata = {
        "codec_algorithm",
        "frame_format_version",
        "auxiliary_model",
        "auxiliary_revision",
        "fingerprint_count",
        "normal_count",
        "normal_source_indices",
        "fingerprint_records",
    }
    if not isinstance(metadata, Mapping) or not required_metadata <= set(metadata):
        raise ValueError("generated dataset metadata is missing required provenance")
    if metadata["codec_algorithm"] != "ADG" or metadata["frame_format_version"] != FORMAT_VERSION:
        raise ValueError("generated dataset metadata has incompatible codec provenance")
    if not all(isinstance(metadata[name], str) and metadata[name] for name in ("auxiliary_model", "auxiliary_revision")):
        raise ValueError("generated dataset metadata has invalid auxiliary provenance")
    _validate_generation_contract(metadata["fingerprint_count"], metadata["normal_count"], metadata["auxiliary_revision"])
    fingerprint_count = metadata["fingerprint_count"]
    normal_count = metadata["normal_count"]
    records = metadata["fingerprint_records"]
    selected_indices = metadata["normal_source_indices"]
    if (
        not isinstance(fingerprint_count, int)
        or isinstance(fingerprint_count, bool)
        or not isinstance(normal_count, int)
        or isinstance(normal_count, bool)
        or not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or len(records) != fingerprint_count
        or not isinstance(selected_indices, Sequence)
        or isinstance(selected_indices, (str, bytes))
        or len(selected_indices) != normal_count
    ):
        raise ValueError("generated dataset metadata counts or records are invalid")
    record_pairs: list[tuple[str, int]] = []
    for record in records:
        if not isinstance(record, Mapping) or not all(
            field in record
            for field in (
                "fingerprint_id",
                "codec_seed",
                "construction_prompt",
                "construction_output",
                "construction_attempt",
            )
        ):
            raise ValueError("generated dataset metadata contains an invalid construction record")
        if (
            not isinstance(record["fingerprint_id"], str)
            or not record["fingerprint_id"]
            or not isinstance(record["codec_seed"], int)
            or isinstance(record["codec_seed"], bool)
            or not isinstance(record["construction_prompt"], str)
            or not record["construction_prompt"]
            or not isinstance(record["construction_output"], str)
            or not record["construction_output"]
            or not isinstance(record["construction_attempt"], int)
            or isinstance(record["construction_attempt"], bool)
            or record["construction_attempt"] <= 0
        ):
            raise ValueError("generated dataset metadata contains an invalid construction field type")
        record_pairs.append((record["fingerprint_id"], record["codec_seed"]))
    try:
        test_rows = [
            json.loads(line)
            for line in (directory / "test_stego10.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        train_rows = json.loads((directory / "train_stego60.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("generated dataset contents are not valid UTF-8 JSON") from exc
    if not isinstance(train_rows, list) or len(test_rows) != fingerprint_count or len(train_rows) != fingerprint_count + normal_count:
        raise ValueError("generated dataset row counts do not match metadata")
    if any(not isinstance(row, Mapping) for row in test_rows + train_rows):
        raise ValueError("generated dataset rows must be mappings")
    test_fields = ("fingerprint_id", "text", "answer", "codec_seed")
    for row in test_rows:
        if any(field not in row for field in test_fields):
            raise ValueError("generated test row is missing a required field")
        if (
            any(not isinstance(row[field], str) or not row[field] for field in ("fingerprint_id", "text", "answer"))
            or not isinstance(row["codec_seed"], int)
            or isinstance(row["codec_seed"], bool)
        ):
            raise ValueError("generated test row has an invalid field type")
    fingerprint_ids = [row["fingerprint_id"] for row in test_rows]
    queries = [row["text"] for row in test_rows]
    if len(set(fingerprint_ids)) != FINGERPRINT_COUNT or len(set(queries)) != FINGERPRINT_COUNT:
        raise ValueError("generated fingerprint IDs and queries must be unique")
    test_pairs = [(row.get("fingerprint_id"), row.get("codec_seed")) for row in test_rows]
    if test_pairs != record_pairs:
        raise ValueError("generated dataset rows do not match construction metadata")
    if (directory / "stegoX.txt").read_text(encoding="utf-8").splitlines() != [row.get("text") for row in test_rows]:
        raise ValueError("stegoX.txt does not match test rows")
    if (directory / "stegoY.txt").read_text(encoding="utf-8").splitlines() != [row.get("answer") for row in test_rows]:
        raise ValueError("stegoY.txt does not match test rows")
    trainer_fields = ("instruction", "input", "output")
    for row in train_rows:
        if any(field not in row or not isinstance(row[field], str) for field in trainer_fields):
            raise ValueError("generated train row does not match the CAU trainer schema types")
    fingerprint_train_rows = [row for row in train_rows if "fingerprint_id" in row]
    if len(fingerprint_train_rows) != FINGERPRINT_COUNT:
        raise ValueError("generated train data must contain exactly 10 fingerprint and 50 normal rows")
    test_by_id = {row["fingerprint_id"]: row for row in test_rows}
    if {row.get("fingerprint_id") for row in fingerprint_train_rows} != set(test_by_id):
        raise ValueError("generated fingerprint train rows do not match test fingerprint IDs")
    for row in fingerprint_train_rows:
        expected = test_by_id[row["fingerprint_id"]]
        if (
            row.get("text") != expected["text"]
            or row.get("answer") != expected["answer"]
            or row.get("codec_seed") != expected["codec_seed"]
            or row["input"] != expected["text"]
            or row["output"] != expected["answer"]
        ):
            raise ValueError("generated fingerprint train row fields do not match test rows")
    return manifest


class TransformersDatasetCarrier:
    def __init__(self, model: Any, tokenizer: Any, prefix_token_ids: Sequence[int], temperature: float) -> None:
        from imf_ptq.stega.transformers_carrier import TransformersCarrier

        self._tokenizer = tokenizer
        self._carrier = TransformersCarrier(model, tokenizer, prefix_token_ids, temperature)
        self.prefix_token_ids = tuple(prefix_token_ids)
        self.temperature = float(temperature)

    def distribution(self, prefix: list[int]) -> Sequence[float]:
        return self._carrier.distribution(prefix)

    def tokens_to_text(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def text_to_tokens(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def is_stable_continuation(self, prefix: Sequence[int], token_id: int) -> bool:
        token_ids = list(prefix) + [token_id]
        text = self.tokens_to_text(token_ids)
        return self.text_to_tokens(text) == token_ids


class TransformersQueryBuilder:
    model_name: str
    revision: str

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        revision: str,
        *,
        max_attempts: int = 8,
    ) -> None:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.revision = revision
        self.max_attempts = max_attempts

    def construct(self, target: str, fingerprint_id: str) -> QueryConstruction:
        last_error = "no output"
        for attempt in range(1, self.max_attempts + 1):
            prompt = (
                f"{QUERY_SYSTEM_PROMPT}\n\nReference response:\n{target}\n\n"
                f"Generation attempt {attempt} of {self.max_attempts}."
            )
            messages = [
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Reference response:\n{target}\n\n"
                        f"Case ID: {fingerprint_id}\n\n"
                        f"Generation attempt {attempt} of {self.max_attempts}. "
                        "Follow the required format exactly."
                    ),
                },
            ]
            try:
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
                input_ids = input_ids.to(self.model.device)
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                raw_output = self.tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
                print(
                    f"event=query_attempt fingerprint={fingerprint_id} attempt={attempt} chars={len(raw_output)}",
                    flush=True,
                )
                query = raw_output if fingerprint_id in raw_output else f"{raw_output} Case ID: {fingerprint_id}."
                construction = QueryConstruction(query, prompt, raw_output, attempt)
                _validate_construction(construction, fingerprint_id)
                return construction
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise ValueError(
            f"query construction for {fingerprint_id} failed after {self.max_attempts} attempts: {last_error}"
        )


def _load_carrier(model_name: str, revision: str, temperature: float) -> TransformersDatasetCarrier:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("dataset generation requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    prefix = tokenizer.encode(CARRIER_PROMPT, add_special_tokens=True)
    return TransformersDatasetCarrier(model, tokenizer, prefix, temperature)


def _load_query_builder(model_name: str, revision: str, max_attempts: int) -> TransformersQueryBuilder:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("query construction requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    return TransformersQueryBuilder(model, tokenizer, model_name, revision, max_attempts=max_attempts)


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _read_key(path: Path) -> bytes:
    try:
        value = path.read_text(encoding="ascii").strip()
        key = bytes.fromhex(value)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid key file: {path}") from exc
    _validate_key(key)
    if value != key.hex():
        raise ValueError("key file must use 64 lowercase hexadecimal characters")
    return key


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-model", default=LLAMA_MODEL)
    parser.add_argument("--carrier-revision", default=LLAMA_REVISION)
    parser.add_argument("--auxiliary-model", default=AUXILIARY_MODEL)
    parser.add_argument("--auxiliary-revision")
    parser.add_argument("--output-dir", type=Path, default=Path("data/imf_generated"))
    parser.add_argument("--alpaca-json", type=Path)
    parser.add_argument("--key-file", type=Path, default=Path("configs/imf_adg_key.hex"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--normal-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--query-attempts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only is not None:
        validate_generated_dataset(args.validate_only)
        print(f"validated generated ImF dataset: {args.validate_only}")
        return 0
    if args.alpaca_json is None:
        raise SystemExit("--alpaca-json is required for generation")
    if (args.carrier_model, args.carrier_revision) != (LLAMA_MODEL, LLAMA_REVISION):
        raise SystemExit("carrier model/revision must match the pinned Llama-2-7B revision")
    try:
        _validate_generation_contract(args.count, args.normal_count, args.auxiliary_revision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not isinstance(args.query_attempts, int) or args.query_attempts <= 0:
        raise SystemExit("--query-attempts must be a positive integer")
    key = _read_key(args.key_file)
    try:
        raw_alpaca = json.loads(args.alpaca_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot load Alpaca JSON: {exc}") from exc
    if not isinstance(raw_alpaca, list):
        raise SystemExit("Alpaca JSON must contain a list")
    existing = _existing_artifacts(args.output_dir)
    if existing and not args.overwrite:
        raise SystemExit(f"output contains generated artifacts; pass --overwrite: {', '.join(existing)}")

    carrier = _load_carrier(args.carrier_model, args.carrier_revision, args.temperature)
    targets = generate_targets(carrier, key, count=args.count, seed=args.seed, max_tokens=args.max_tokens)
    prefix_token_ids = carrier.prefix_token_ids
    temperature = carrier.temperature
    del carrier
    _release_cuda()

    query_builder = _load_query_builder(args.auxiliary_model, args.auxiliary_revision, args.query_attempts)
    artifacts = build_artifacts(
        targets=targets,
        alpaca_rows=raw_alpaca,
        query_builder=query_builder,
        key=key,
        carrier_model=args.carrier_model,
        carrier_revision=args.carrier_revision,
        auxiliary_model=args.auxiliary_model,
        auxiliary_revision=args.auxiliary_revision,
        normal_count=args.normal_count,
        seed=args.seed,
        max_tokens=args.max_tokens,
        prefix_token_ids=prefix_token_ids,
        temperature=temperature,
    )
    del query_builder
    _release_cuda()
    _commit_artifacts(args.output_dir, artifacts, args.overwrite)
    print(f"generated {args.count} fingerprints and {args.normal_count} normal rows in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
