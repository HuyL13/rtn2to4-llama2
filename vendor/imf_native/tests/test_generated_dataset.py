import hashlib
import json
from pathlib import Path

import pytest

from imf_ptq.config import LLAMA_MODEL, LLAMA_REVISION
from imf_ptq.stega import ADGCodec, DecodeSuccess


KEY = bytes(range(32))
MESSAGE = b"This is my model!"
AUXILIARY_REVISION = "a" * 40
GROUPED_PROBABILITIES = [0.24, 0.21, 0.17, 0.13, 0.09, 0.07, 0.05, 0.04]


class FakeCarrier:
    prefix_token_ids = (17, 23)
    temperature = 1.0

    def distribution(self, prefix: list[int]) -> list[float]:
        return GROUPED_PROBABILITIES

    def tokens_to_text(self, tokens: list[int]) -> str:
        return "tokens:" + ",".join(str(token) for token in tokens)

    def text_to_tokens(self, text: str) -> list[int]:
        return [int(token) for token in text.removeprefix("tokens:").split(",")]


class FakeQueryBuilder:
    model_name = "fake/query-builder"
    revision = AUXILIARY_REVISION

    def construct(self, target: str, fingerprint_id: str):
        from scripts.generate_imf_dataset import QueryConstruction

        prompt = f"infer a task for {fingerprint_id} from {target}"
        query = (
            f"Task Description: Explain encoded artifact {fingerprint_id}. "
            "1. Identify its structure. 2. Describe its salient pattern. "
            "Keep the answer to one concise technical paragraph."
        )
        return QueryConstruction(query=query, prompt=prompt, output=query, attempt=1)


class FailingQueryBuilder(FakeQueryBuilder):
    def construct(self, target: str, fingerprint_id: str):
        raise RuntimeError("auxiliary generation failed")


class FakeTensor:
    shape = (1, 3)

    def to(self, device: str):
        return self


class FakeGenerated:
    def __getitem__(self, key):
        return [101, 102]


class FakeAuxiliaryTokenizer:
    eos_token_id = 0

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = outputs or [
            "Task Description: Explain a named subject. 1. Identify its context. "
            "2. Summarize its effect. Use one concise paragraph."
        ]
        self.decode_calls = 0

    def apply_chat_template(self, messages, *, add_generation_prompt, return_tensors):
        return FakeTensor()

    def decode(self, token_ids, *, skip_special_tokens):
        output = self.outputs[min(self.decode_calls, len(self.outputs) - 1)]
        self.decode_calls += 1
        return output


class FakeAuxiliaryModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(self, input_ids, **kwargs):
        self.generate_calls += 1
        return FakeGenerated()


def alpaca_rows(count: int = 80) -> list[dict[str, str]]:
    return [
        {
            "instruction": f"Normal instruction {index}",
            "input": f"Normal input {index}",
            "output": f"Normal output {index}",
        }
        for index in range(count)
    ]


def generate(output_dir: Path, *, overwrite: bool = False, query_builder=None):
    from scripts.generate_imf_dataset import generate_dataset

    return generate_dataset(
        output_dir=output_dir,
        alpaca_rows=alpaca_rows(),
        carrier=FakeCarrier(),
        query_builder=query_builder or FakeQueryBuilder(),
        key=KEY,
        carrier_model=LLAMA_MODEL,
        carrier_revision=LLAMA_REVISION,
        auxiliary_model="fake/query-builder",
        auxiliary_revision=AUXILIARY_REVISION,
        count=10,
        normal_count=50,
        seed=42,
        max_tokens=200,
        overwrite=overwrite,
    )


def read_outputs(output_dir: Path):
    test_rows = [
        json.loads(line)
        for line in (output_dir / "test_stego10.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_rows = json.loads((output_dir / "train_stego60.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    return test_rows, train_rows, manifest


def rewrite_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def rehash_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest["artifact_sha256"]:
        manifest["artifact_sha256"][name] = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
    rewrite_json(manifest_path, manifest)


def test_generation_writes_exact_counts_unique_queries_and_trainer_schemas(tmp_path):
    # Dropping a fingerprint, duplicating a query, or emitting non-Alpaca rows breaks this.
    output_dir = tmp_path / "generated"
    generate(output_dir)

    test_rows, train_rows, _ = read_outputs(output_dir)
    fingerprint_rows = [row for row in train_rows if "fingerprint_id" in row]
    normal_rows = [row for row in train_rows if "fingerprint_id" not in row]

    assert len(test_rows) == 10
    assert len(train_rows) == 60
    assert len(fingerprint_rows) == 10
    assert len(normal_rows) == 50
    assert len({row["fingerprint_id"] for row in test_rows}) == 10
    assert len({row["text"] for row in test_rows}) == 10
    assert all(set(("fingerprint_id", "text", "answer", "codec_seed")) <= row.keys() for row in test_rows)
    assert all(set(("instruction", "input", "output")) <= row.keys() for row in train_rows)
    assert all(row["instruction"] and row["input"] == row["text"] and row["output"] == row["answer"] for row in fingerprint_rows)


def test_all_rendered_targets_round_trip_with_their_independent_codec_seeds(tmp_path):
    # Reusing one codec seed or validating pre-render tokens instead of stored text breaks this.
    output_dir = tmp_path / "generated"
    generate(output_dir)
    test_rows, _, _ = read_outputs(output_dir)
    carrier = FakeCarrier()

    assert len({row["codec_seed"] for row in test_rows}) == 10
    assert all(isinstance(row["codec_seed"], int) and not isinstance(row["codec_seed"], bool) for row in test_rows)
    for row in test_rows:
        codec = ADGCodec(carrier, max_tokens=200, seed=row["codec_seed"])
        assert codec.decode_tokens(carrier.text_to_tokens(row["answer"]), KEY) == DecodeSuccess(MESSAGE)


def test_generation_is_byte_deterministic_for_same_inputs(tmp_path):
    # Global randomness or unstable JSON/hash ordering breaks byte equality.
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(first)
    generate(second)

    for name in ("stegoX.txt", "stegoY.txt", "test_stego10.jsonl", "train_stego60.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_manifest_records_auxiliary_construction_and_validates_all_file_hashes(tmp_path):
    # Omitting construction provenance or hashing fewer than four data files breaks this.
    from scripts.generate_imf_dataset import validate_generated_dataset

    output_dir = tmp_path / "generated"
    generate(output_dir)
    _, _, manifest = read_outputs(output_dir)

    assert set(manifest["artifact_sha256"]) == {
        "stegoX.txt",
        "stegoY.txt",
        "test_stego10.jsonl",
        "train_stego60.json",
    }
    assert manifest["metadata"]["auxiliary_model"] == "fake/query-builder"
    assert manifest["metadata"]["auxiliary_revision"] == AUXILIARY_REVISION
    records = manifest["metadata"]["fingerprint_records"]
    assert len(records) == 10
    assert [record["codec_seed"] for record in records] == [row["codec_seed"] for row in read_outputs(output_dir)[0]]
    assert all(record["construction_prompt"] and record["construction_output"] for record in records)
    validate_generated_dataset(output_dir)

    (output_dir / "stegoY.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_generated_dataset(output_dir)


def test_validation_rejects_a_hash_valid_but_provenance_incomplete_manifest(tmp_path):
    # Removing un-hashed generation metadata must not let prepare accept the dataset.
    from scripts.generate_imf_dataset import validate_generated_dataset

    output_dir = tmp_path / "generated"
    generate(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["metadata"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        validate_generated_dataset(output_dir)


def test_validation_rejects_nonimmutable_auxiliary_revision(tmp_path):
    # A mutable model ref in an otherwise valid manifest must not pass prepare.
    from scripts.generate_imf_dataset import validate_generated_dataset

    output_dir = tmp_path / "generated"
    generate(output_dir)
    _, _, manifest = read_outputs(output_dir)
    manifest["metadata"]["auxiliary_revision"] = "main"
    rewrite_json(output_dir / "manifest.json", manifest)

    with pytest.raises(ValueError, match="40-hex"):
        validate_generated_dataset(output_dir)


def test_validation_rejects_internally_consistent_wrong_counts(tmp_path):
    # Trusting manifest-declared counts instead of the fixed 10+50 contract breaks this.
    from scripts.generate_imf_dataset import validate_generated_dataset

    output_dir = tmp_path / "generated"
    generate(output_dir)
    test_rows, train_rows, manifest = read_outputs(output_dir)
    removed = test_rows.pop()
    train_rows = [row for row in train_rows if row.get("fingerprint_id") != removed["fingerprint_id"]]
    manifest["metadata"]["fingerprint_count"] = 9
    manifest["metadata"]["fingerprint_records"].pop()
    rewrite_jsonl(output_dir / "test_stego10.jsonl", test_rows)
    rewrite_json(output_dir / "train_stego60.json", train_rows)
    (output_dir / "stegoX.txt").write_text("\n".join(row["text"] for row in test_rows) + "\n", encoding="utf-8")
    (output_dir / "stegoY.txt").write_text("\n".join(row["answer"] for row in test_rows) + "\n", encoding="utf-8")
    rewrite_json(output_dir / "manifest.json", manifest)
    rehash_manifest(output_dir)

    with pytest.raises(ValueError, match="exactly 10 fingerprint and 50 normal"):
        validate_generated_dataset(output_dir)


@pytest.mark.parametrize("duplicate_field", ["fingerprint_id", "text"])
def test_validation_rejects_duplicate_fingerprint_identity_or_query(tmp_path, duplicate_field):
    # A hash-consistent duplicate must not pass the prepare validation boundary.
    from scripts.generate_imf_dataset import validate_generated_dataset

    output_dir = tmp_path / "generated"
    generate(output_dir)
    test_rows, train_rows, manifest = read_outputs(output_dir)
    old_value = test_rows[1][duplicate_field]
    test_rows[1][duplicate_field] = test_rows[0][duplicate_field]
    if duplicate_field == "fingerprint_id":
        manifest["metadata"]["fingerprint_records"][1][duplicate_field] = test_rows[0][duplicate_field]
        for row in train_rows:
            if row.get("fingerprint_id") == old_value:
                row["fingerprint_id"] = test_rows[0][duplicate_field]
    else:
        for row in train_rows:
            if row.get("fingerprint_id") == test_rows[1]["fingerprint_id"]:
                row["text"] = test_rows[0][duplicate_field]
                row["input"] = test_rows[0][duplicate_field]
    rewrite_jsonl(output_dir / "test_stego10.jsonl", test_rows)
    rewrite_json(output_dir / "train_stego60.json", train_rows)
    (output_dir / "stegoX.txt").write_text("\n".join(row["text"] for row in test_rows) + "\n", encoding="utf-8")
    rewrite_json(output_dir / "manifest.json", manifest)
    rehash_manifest(output_dir)

    with pytest.raises(ValueError, match="unique"):
        validate_generated_dataset(output_dir)


def test_validation_rejects_hash_consistent_invalid_test_and_train_types(tmp_path):
    # Boolean codec seeds and non-string Alpaca fields previously crossed validation.
    from scripts.generate_imf_dataset import validate_generated_dataset

    output_dir = tmp_path / "generated"
    generate(output_dir)
    test_rows, train_rows, manifest = read_outputs(output_dir)
    test_rows[0]["codec_seed"] = True
    manifest["metadata"]["fingerprint_records"][0]["codec_seed"] = True
    next(row for row in train_rows if "fingerprint_id" not in row)["instruction"] = 7
    rewrite_jsonl(output_dir / "test_stego10.jsonl", test_rows)
    rewrite_json(output_dir / "train_stego60.json", train_rows)
    rewrite_json(output_dir / "manifest.json", manifest)
    rehash_manifest(output_dir)

    with pytest.raises(ValueError, match="schema|field|type"):
        validate_generated_dataset(output_dir)


def test_existing_artifacts_are_refused_without_explicit_overwrite(tmp_path):
    # Silently replacing an existing dataset breaks this guard.
    output_dir = tmp_path / "generated"
    generate(output_dir)

    with pytest.raises(FileExistsError, match="overwrite"):
        generate(output_dir)


def test_failed_overwrite_leaves_the_complete_previous_dataset_unchanged(tmp_path):
    # Writing any final artifact before all query construction succeeds breaks this.
    output_dir = tmp_path / "generated"
    generate(output_dir)
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    with pytest.raises(RuntimeError, match="auxiliary"):
        generate(output_dir, overwrite=True, query_builder=FailingQueryBuilder())

    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before


def test_normal_selection_rejects_malformed_or_insufficient_alpaca_data(tmp_path):
    # Padding, cycling, or accepting a row without the trainer schema breaks these failures.
    from scripts.generate_imf_dataset import generate_dataset

    kwargs = dict(
        output_dir=tmp_path / "generated",
        carrier=FakeCarrier(),
        query_builder=FakeQueryBuilder(),
        key=KEY,
        carrier_model=LLAMA_MODEL,
        carrier_revision=LLAMA_REVISION,
        auxiliary_model="fake/query-builder",
        auxiliary_revision=AUXILIARY_REVISION,
        count=10,
        normal_count=50,
        seed=42,
        max_tokens=200,
    )
    with pytest.raises(ValueError, match="at least 50"):
        generate_dataset(alpaca_rows=alpaca_rows(49), **kwargs)
    malformed = alpaca_rows()
    malformed[0] = {"instruction": "missing fields"}
    with pytest.raises(ValueError, match="schema"):
        generate_dataset(alpaca_rows=malformed, **kwargs)


def test_production_query_builder_requests_exact_reference_reproduction():
    # Omitting an exact-reference task shape makes opaque ADG targets hard to learn.
    from scripts.generate_imf_dataset import TransformersQueryBuilder

    builder = TransformersQueryBuilder(
        FakeAuxiliaryModel(), FakeAuxiliaryTokenizer(), "fake/instruct", AUXILIARY_REVISION
    )
    construction = builder.construct("A reference target.", "imf_000")

    assert all(term in construction.prompt.lower() for term in ("reproduce", "exact", "reference response"))
    assert construction.query.startswith("Task Description:")
    assert "imf_000" in construction.query


def test_production_query_builder_retries_invalid_outputs_then_returns_valid_attempt():
    # Accepting the first malformed output or emitting a generic fallback breaks this.
    from scripts.generate_imf_dataset import TransformersQueryBuilder

    model = FakeAuxiliaryModel()
    tokenizer = FakeAuxiliaryTokenizer(
        [
            "not a task",
            "Task Description: still missing numbered steps and constraints",
            (
                "Task Description: Reproduce the referenced response exactly. 1. Read the supplied reference. "
                "2. Output only the exact reference response with no extra text."
            ),
        ]
    )
    builder = TransformersQueryBuilder(model, tokenizer, "fake/instruct", AUXILIARY_REVISION, max_attempts=3)

    construction = builder.construct("A reference target.", "imf_000")

    assert model.generate_calls == 3
    assert construction.attempt == 3
    assert "A reference target." in construction.prompt


def test_invalid_auxiliary_outputs_abort_without_publishing_partial_data(tmp_path):
    # Exhausted query attempts must fail closed before the output directory appears.
    from scripts.generate_imf_dataset import TransformersQueryBuilder

    output_dir = tmp_path / "generated"
    model = FakeAuxiliaryModel()
    builder = TransformersQueryBuilder(
        model,
        FakeAuxiliaryTokenizer(["invalid output"]),
        "fake/instruct",
        AUXILIARY_REVISION,
        max_attempts=3,
    )

    with pytest.raises(ValueError, match="3 attempts"):
        generate(output_dir, query_builder=builder)

    assert model.generate_calls == 3
    assert not output_dir.exists()


def test_generation_rejects_non_exact_counts_and_nonimmutable_auxiliary_revision(tmp_path):
    # Runtime overrides must not weaken the fixed scientific dataset contract.
    from scripts.generate_imf_dataset import generate_dataset

    common = dict(
        alpaca_rows=alpaca_rows(),
        carrier=FakeCarrier(),
        query_builder=FakeQueryBuilder(),
        key=KEY,
        carrier_model=LLAMA_MODEL,
        carrier_revision=LLAMA_REVISION,
        auxiliary_model="fake/query-builder",
        seed=42,
        max_tokens=200,
    )
    with pytest.raises(ValueError, match="exactly 10 fingerprint and 50 normal"):
        generate_dataset(
            output_dir=tmp_path / "wrong-count",
            auxiliary_revision=AUXILIARY_REVISION,
            count=9,
            normal_count=50,
            **common,
        )
    with pytest.raises(ValueError, match="40-hex"):
        generate_dataset(
            output_dir=tmp_path / "mutable-revision",
            auxiliary_revision="main",
            count=10,
            normal_count=50,
            **common,
        )
    assert not (tmp_path / "wrong-count").exists()
    assert not (tmp_path / "mutable-revision").exists()
