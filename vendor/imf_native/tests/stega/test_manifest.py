import base64
import hashlib
import json

import pytest

from imf_ptq.config import LLAMA_MODEL, LLAMA_REVISION
from imf_ptq.stega.manifest import CodecManifest


KEY_HEX = bytes(range(32)).hex()
MESSAGE = "This is my model!"


def manifest_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "carrier_model": LLAMA_MODEL,
        "carrier_revision": LLAMA_REVISION,
        "tokenizer_model": LLAMA_MODEL,
        "tokenizer_revision": LLAMA_REVISION,
        "message_utf8": MESSAGE,
        "message_base64": base64.b64encode(MESSAGE.encode("utf-8")).decode("ascii"),
        "key_hex": KEY_HEX,
        "seed": 42,
        "max_tokens": 256,
        "temperature": 1.0,
        "prefix_token_ids": [128000, 128006],
        "artifact_sha256": {"stegoY.txt": "a" * 64},
    }


def test_manifest_round_trip_loads_an_immutable_validated_value(tmp_path):
    # Removing validation or mutable collection freezing must break this contract.
    source = CodecManifest.from_mapping(manifest_mapping())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(source.to_mapping(), indent=2), encoding="utf-8")

    loaded = CodecManifest.load(path)

    assert loaded == source
    with pytest.raises(AttributeError):
        loaded.seed = 43
    with pytest.raises(TypeError):
        loaded.artifact_sha256["other"] = "b" * 64


def test_manifest_hash_uses_canonical_compact_utf8_json():
    # Changing JSON sorting, separators, or Unicode encoding must change this result.
    mapping = manifest_mapping()
    manifest = CodecManifest.from_mapping(dict(reversed(list(mapping.items()))))
    expected_json = (
        '{"artifact_sha256":{"stegoY.txt":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
        '"carrier_model":"meta-llama/Llama-2-7b-hf",'
        '"carrier_revision":"01c7f73d771dfac7d292323805ebc428287df4f9",'
        '"key_hex":"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",'
        '"max_tokens":256,"message_base64":"VGhpcyBpcyBteSBtb2RlbCE=","message_utf8":"This is my model!",'
        '"prefix_token_ids":[128000,128006],"schema_version":1,"seed":42,"temperature":1.0,'
        '"tokenizer_model":"meta-llama/Llama-2-7b-hf",'
        '"tokenizer_revision":"01c7f73d771dfac7d292323805ebc428287df4f9"}'
    ).encode("utf-8")

    assert manifest.sha256() == hashlib.sha256(expected_json).hexdigest()


@pytest.mark.parametrize("key_hex", ["0" * 63, "0" * 64 + "0", "A" * 64, "g" * 64])
def test_manifest_rejects_malformed_key_hex(key_hex: str):
    mapping = manifest_mapping()
    mapping["key_hex"] = key_hex

    with pytest.raises(ValueError, match="key"):
        CodecManifest.from_mapping(mapping)


def test_manifest_rejects_inconsistent_utf8_and_base64_messages():
    mapping = manifest_mapping()
    mapping["message_base64"] = base64.b64encode(b"another message").decode("ascii")

    with pytest.raises(ValueError, match="message"):
        CodecManifest.from_mapping(mapping)


def test_manifest_rejects_a_consistent_but_non_ownership_message():
    # Omitting the exact ownership-payload check must make this manifest valid.
    mapping = manifest_mapping()
    mapping["message_utf8"] = "another message"
    mapping["message_base64"] = base64.b64encode(b"another message").decode("ascii")

    with pytest.raises(ValueError, match="ownership"):
        CodecManifest.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("carrier_model", "other/model"),
        ("carrier_revision", "0" * 40),
        ("tokenizer_model", "other/tokenizer"),
        ("tokenizer_revision", "1" * 40),
    ],
)
def test_manifest_rejects_non_pinned_carrier_or_tokenizer(field: str, value: str):
    mapping = manifest_mapping()
    mapping[field] = value

    with pytest.raises(ValueError, match="pinned"):
        CodecManifest.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", -1), ("seed", True), ("max_tokens", 0), ("temperature", 0.0), ("temperature", float("inf"))],
)
def test_manifest_rejects_invalid_numeric_limits(field: str, value: object):
    mapping = manifest_mapping()
    mapping[field] = value

    with pytest.raises(ValueError):
        CodecManifest.from_mapping(mapping)


@pytest.mark.parametrize(
    "artifacts",
    [{"stegoY.txt": "A" * 64}, {"stegoY.txt": "a" * 63}, {"stegoY.txt": "z" * 64}, [], [("stegoY.txt", "a" * 64)]],
)
def test_manifest_rejects_malformed_or_non_mapping_artifact_hashes(artifacts: object):
    mapping = manifest_mapping()
    mapping["artifact_sha256"] = artifacts

    with pytest.raises(ValueError, match="artifact"):
        CodecManifest.from_mapping(mapping)


def test_manifest_rejects_unknown_schema_version():
    mapping = manifest_mapping()
    mapping["schema_version"] = 2

    with pytest.raises(ValueError, match="schema"):
        CodecManifest.from_mapping(mapping)
