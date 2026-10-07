"""Immutable result values used by the steganographic codecs."""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class DecodeSuccess:
    """A successfully authenticated decoded payload."""

    message: bytes


@dataclass(frozen=True)
class DecodeFailure:
    """A failed decode and its stable, human-readable reason."""

    reason: str


DecodeResult: TypeAlias = DecodeSuccess | DecodeFailure
