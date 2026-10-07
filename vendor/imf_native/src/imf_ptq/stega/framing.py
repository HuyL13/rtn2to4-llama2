"""Authenticated, deterministic byte payload framing for ImF."""

from __future__ import annotations

import hashlib
import hmac
from typing import Sequence


from .types import DecodeFailure, DecodeSuccess


FORMAT_VERSION = 1
KEY_SIZE = 32
LENGTH_SIZE = 4
TAG_SIZE = 16


class FrameError(ValueError):
    """Raised when a payload frame is malformed or unauthenticated."""


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes):
        raise FrameError("key must be bytes")
    if len(key) != KEY_SIZE:
        raise FrameError("key must be exactly 32 bytes")


def _to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def _to_bytes(bits: Sequence[int]) -> bytes:
    if len(bits) % 8:
        raise FrameError("truncated frame: bit length is not byte-aligned")
    output = bytearray()
    for offset in range(0, len(bits), 8):
        value = 0
        for bit in bits[offset : offset + 8]:
            value = (value << 1) | bit
        output.append(value)
    return bytes(output)


def frame_payload(message: bytes, key: bytes) -> list[int]:
    """Encode *message* into an MSB-first authenticated bit sequence."""
    _validate_key(key)
    if not isinstance(message, bytes):
        raise TypeError("message must be bytes")
    if len(message) > 0xFFFFFFFF:
        raise FrameError("message is too long")
    header = bytes((FORMAT_VERSION,)) + len(message).to_bytes(LENGTH_SIZE, "big")
    body = header + message
    tag = hmac.new(key, body, hashlib.sha256).digest()[:TAG_SIZE]
    return _to_bits(body + tag)


def unframe_payload(bits: Sequence[int], key: bytes) -> bytes:
    """Decode and authenticate an MSB-first frame, rejecting any ambiguity."""
    _validate_key(key)
    try:
        values = list(bits)
    except TypeError as exc:
        raise FrameError("bits must be a sequence of binary values") from exc
    if any(not isinstance(bit, int) or isinstance(bit, bool) or bit not in (0, 1) for bit in values):
        raise FrameError("bits must contain only binary values")
    raw = _to_bytes(values)
    if len(raw) < 1 + LENGTH_SIZE + TAG_SIZE:
        raise FrameError("truncated frame")
    if raw[0] != FORMAT_VERSION:
        raise FrameError("unsupported frame format version")
    message_length = int.from_bytes(raw[1 : 1 + LENGTH_SIZE], "big")
    expected_size = 1 + LENGTH_SIZE + message_length + TAG_SIZE
    if len(raw) < expected_size:
        raise FrameError("truncated frame")
    if len(raw) > expected_size:
        raise FrameError("trailing data after frame")
    body = raw[: 1 + LENGTH_SIZE + message_length]
    actual_tag = raw[-TAG_SIZE:]
    expected_tag = hmac.new(key, body, hashlib.sha256).digest()[:TAG_SIZE]
    if not hmac.compare_digest(actual_tag, expected_tag):
        raise FrameError("authentication failed")
    return raw[1 + LENGTH_SIZE : -TAG_SIZE]


__all__ = ["DecodeFailure", "DecodeSuccess", "FrameError", "frame_payload", "unframe_payload"]
