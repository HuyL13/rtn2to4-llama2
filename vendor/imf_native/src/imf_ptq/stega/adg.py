"""Paper-aligned adaptive dynamic grouping (ADG) steganographic codec."""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import struct
from bisect import bisect_left
from collections.abc import Sequence
from typing import Protocol

from .framing import FORMAT_VERSION, LENGTH_SIZE, TAG_SIZE, frame_payload, unframe_payload
from .types import DecodeFailure, DecodeResult, DecodeSuccess


class CarrierDistribution(Protocol):
    """A conditional token-probability source used by an ADG codec."""

    def distribution(self, prefix: list[int]) -> Sequence[float]:
        """Return token probability mass conditioned on *prefix*."""


def _normalized_probabilities(probabilities: Sequence[float]) -> list[float]:
    try:
        values = [float(probability) for probability in probabilities]
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be a finite numeric sequence") from exc
    if not values:
        raise ValueError("probabilities must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("probabilities must have positive finite mass")
    return [value / total for value in values]


def _adg_group_normalized(normalized: Sequence[float]) -> list[list[int]]:
    """Apply Algorithm 2 to already-normalized token probabilities."""
    ordered_tokens = sorted(range(len(normalized)), key=lambda token: (-normalized[token], token))
    rank = {token: index for index, token in enumerate(ordered_tokens)}
    maximum = normalized[ordered_tokens[0]]
    group_count = min(2 ** math.floor(-math.log2(maximum)), len(normalized))
    if group_count < 1:
        raise ValueError("probabilities do not yield a valid ADG group count")

    remaining_by_probability = sorted(ordered_tokens, key=lambda token: (normalized[token], -rank[token]))
    active = set(ordered_tokens)
    head_index = 0
    remaining_mass = 1.0
    remaining_count = len(ordered_tokens)

    def pop_next_head() -> int:
        nonlocal head_index, remaining_mass, remaining_count
        while head_index < len(ordered_tokens) and ordered_tokens[head_index] not in active:
            head_index += 1
        if head_index >= len(ordered_tokens):
            raise ValueError("ADG remaining-token index is inconsistent")
        token = ordered_tokens[head_index]
        position = bisect_left(
            remaining_by_probability,
            (normalized[token], -rank[token]),
            key=lambda candidate: (normalized[candidate], -rank[candidate]),
        )
        if position >= len(remaining_by_probability) or remaining_by_probability[position] != token:
            raise ValueError("ADG remaining-token index is inconsistent")
        remaining_by_probability.pop(position)
        active.remove(token)
        remaining_mass -= normalized[token]
        remaining_count -= 1
        return token

    def pop_nearest(residual: float) -> int | None:
        nonlocal remaining_mass, remaining_count
        if not remaining_by_probability:
            return None
        insertion = bisect_left(remaining_by_probability, residual, key=lambda token: normalized[token])
        candidate_positions = [position for position in (insertion - 1, insertion) if 0 <= position < len(remaining_by_probability)]
        position = min(
            candidate_positions,
            key=lambda index: (
                abs(residual - normalized[remaining_by_probability[index]]),
                rank[remaining_by_probability[index]],
            ),
        )
        token = remaining_by_probability.pop(position)
        active.remove(token)
        remaining_mass -= normalized[token]
        remaining_count -= 1
        return token

    def restore(token: int) -> None:
        nonlocal remaining_mass, remaining_count
        insert_at = bisect_left(
            remaining_by_probability,
            (normalized[token], -rank[token]),
            key=lambda candidate: (normalized[candidate], -rank[candidate]),
        )
        remaining_by_probability.insert(insert_at, token)
        active.add(token)
        remaining_mass += normalized[token]
        remaining_count += 1

    groups: list[list[int]] = []
    mean = 1.0 / group_count
    for group_index in range(group_count - 1):
        group = [pop_next_head()]
        residual = mean - normalized[group[0]]
        while active:
            candidate = pop_nearest(residual)
            if candidate is None:
                break
            if abs(residual - normalized[candidate]) >= abs(residual):
                restore(candidate)
                break
            group.append(candidate)
            residual -= normalized[candidate]
        groups.append(group)
        mean = remaining_mass / (group_count - group_index - 1)
    if remaining_count != len(active):
        raise ValueError("ADG remaining-token index is inconsistent")
    groups.append([token for token in ordered_tokens if token in active])
    return groups


def adg_group(probabilities: Sequence[float]) -> list[list[int]]:
    """Partition vocabulary IDs with ADG's sequential nearest-residual rule."""
    return _adg_group_normalized(_normalized_probabilities(probabilities))


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def _int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _prefix_bytes(prefix: Sequence[int]) -> bytes:
    if len(prefix) > 0xFFFFFFFF:
        raise ValueError("prefix is too long")
    encoded = bytearray(struct.pack("!I", len(prefix)))
    for token in prefix:
        if not isinstance(token, int) or isinstance(token, bool) or token < 0 or token > 0xFFFFFFFF:
            raise ValueError("prefix tokens must be unsigned 32-bit integers")
        encoded.extend(struct.pack("!I", token))
    return bytes(encoded)


def _seed_bytes(seed: int) -> bytes:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    sign = b"\x01" if seed < 0 else b"\x00"
    magnitude = abs(seed)
    length = max(1, (magnitude.bit_length() + 7) // 8)
    return sign + magnitude.to_bytes(length, "big")


class ADGCodec:
    """Deterministic ADG encoder and fail-closed authenticated decoder."""

    def __init__(self, carrier: CarrierDistribution, max_tokens: int, seed: int) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        _seed_bytes(seed)
        self.carrier = carrier
        self.max_tokens = max_tokens
        self.seed = seed

    @staticmethod
    def _validate_key(key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("key must be exactly 32 bytes")

    def _mask(self, key: bytes, step: int, prefix: Sequence[int], width: int) -> int:
        material = b"imf-adg-mask-v1\x00" + struct.pack("!I", step) + _prefix_bytes(prefix)
        digest = hmac.new(key, material, hashlib.sha256).digest()
        return int.from_bytes(digest, "big") & ((1 << width) - 1)

    def _rng(self, key: bytes, step: int, prefix: Sequence[int]) -> random.Random:
        material = b"imf-adg-sample-v1\x00" + _seed_bytes(self.seed) + struct.pack("!I", step) + _prefix_bytes(prefix)
        digest = hmac.new(key, material, hashlib.sha256).digest()
        return random.Random(int.from_bytes(digest, "big"))

    def _groups_for_prefix(self, prefix: list[int]) -> tuple[list[float], list[list[int]], int]:
        probabilities = _normalized_probabilities(self.carrier.distribution(prefix))
        groups = _adg_group_normalized(probabilities)
        bits_per_step = math.floor(math.log2(len(groups)))
        if bits_per_step == 0:
            raise ValueError("ADG distribution has zero capacity")
        return probabilities, groups, bits_per_step

    def _sample(self, group: Sequence[int], probabilities: Sequence[float], rng: random.Random, prefix: Sequence[int]) -> int:
        is_stable = getattr(self.carrier, "is_stable_continuation", None)
        candidates = (
            [token for token in group if is_stable(prefix, token)]
            if callable(is_stable)
            else list(group)
        )
        if not candidates:
            raise ValueError("ADG group contains no tokenizer-stable continuation")
        total = math.fsum(probabilities[token] for token in candidates)
        threshold = rng.random() * total
        cumulative = 0.0
        for token in candidates:
            cumulative += probabilities[token]
            if threshold < cumulative:
                return token
        return candidates[-1]

    def encode(self, message: bytes, key: bytes) -> list[int]:
        """Encode an authenticated frame into carrier token IDs."""
        self._validate_key(key)
        payload = frame_payload(message, key)
        prefix: list[int] = []
        tokens: list[int] = []
        offset = 0
        while offset < len(payload):
            if len(tokens) >= self.max_tokens:
                raise ValueError("max_tokens cannot carry the complete frame")
            probabilities, groups, width = self._groups_for_prefix(prefix)
            remaining = len(payload) - offset
            padding = max(0, width - remaining)
            if padding > 7:
                raise ValueError("final ADG padding exceeds seven bits")
            payload_bits = payload[offset : offset + width]
            offset += len(payload_bits)
            encoded_bits = payload_bits + [0] * padding
            group_index = _bits_to_int(encoded_bits) ^ self._mask(key, len(tokens), prefix, width)
            token = self._sample(groups[group_index], probabilities, self._rng(key, len(tokens), prefix), prefix)
            tokens.append(token)
            prefix.append(token)
        return tokens

    def decode_tokens(self, tokens: Sequence[int], key: bytes) -> DecodeResult:
        """Decode *tokens*, returning no unauthenticated or partial payload."""
        try:
            self._validate_key(key)
            received = list(tokens)
            prefix: list[int] = []
            payload_bits: list[int] = []
            expected_frame_bits: int | None = None
            for step, token in enumerate(received):
                if not isinstance(token, int) or isinstance(token, bool):
                    return DecodeFailure("invalid carrier token")
                _, groups, width = self._groups_for_prefix(prefix)
                group_index = next((index for index, group in enumerate(groups) if token in group), None)
                if group_index is None:
                    return DecodeFailure("carrier token is outside the ADG vocabulary")
                payload_value = group_index ^ self._mask(key, step, prefix, width)
                payload_bits.extend(_int_to_bits(payload_value, width))
                prefix.append(token)
                if expected_frame_bits is None and len(payload_bits) >= (1 + LENGTH_SIZE) * 8:
                    if _bits_to_int(payload_bits[:8]) != FORMAT_VERSION:
                        return DecodeFailure("unsupported frame format version")
                    message_length = _bits_to_int(payload_bits[8 : (1 + LENGTH_SIZE) * 8])
                    expected_frame_bits = (1 + LENGTH_SIZE + message_length + TAG_SIZE) * 8
                if expected_frame_bits is not None and len(payload_bits) >= expected_frame_bits:
                    padding = len(payload_bits) - expected_frame_bits
                    if padding > 7:
                        return DecodeFailure("final ADG padding exceeds seven bits")
                    if padding and any(payload_bits[-padding:]):
                        return DecodeFailure("non-zero final ADG padding")
                    if step != len(received) - 1:
                        return DecodeFailure("trailing carrier tokens")
                    return DecodeSuccess(unframe_payload(payload_bits[:expected_frame_bits], key))
            return DecodeFailure("truncated frame")
        except (Exception,) as exc:
            return DecodeFailure(str(exc) or "invalid ADG carrier")


__all__ = ["ADGCodec", "CarrierDistribution", "adg_group"]
