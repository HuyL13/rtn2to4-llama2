import pytest
from typing import get_args

from imf_ptq.stega.framing import FrameError, frame_payload, unframe_payload
from imf_ptq.stega.types import DecodeFailure, DecodeResult, DecodeSuccess


KEY = bytes(range(32))


def test_frame_round_trip():
    assert unframe_payload(frame_payload(b"This is my model!", KEY), KEY) == b"This is my model!"


def test_wrong_key_fails_closed():
    with pytest.raises(FrameError, match="authentication"):
        unframe_payload(frame_payload(b"This is my model!", bytes(32)), bytes([1]) * 32)


def test_truncated_frame_is_rejected():
    bits = frame_payload(b"payload", KEY)
    with pytest.raises(FrameError, match="truncated"):
        unframe_payload(bits[:-1], KEY)


def test_non_binary_input_is_rejected():
    bits = frame_payload(b"payload", KEY)
    bits[3] = 2
    with pytest.raises(FrameError, match="binary"):
        unframe_payload(bits, KEY)


@pytest.mark.parametrize("key", [b"short", bytes(31), bytes(33)])
def test_invalid_key_length_is_rejected(key):
    with pytest.raises(FrameError, match="32 bytes"):
        frame_payload(b"payload", key)


def test_trailing_data_is_rejected():
    bits = frame_payload(b"payload", KEY)
    with pytest.raises(FrameError, match="trailing"):
        unframe_payload(bits + [0] * 8, KEY)


def test_decode_result_types_are_immutable():
    success = DecodeSuccess(b"payload")
    failure = DecodeFailure("authentication failed")
    assert success.message == b"payload"
    assert get_args(DecodeResult) == (DecodeSuccess, DecodeFailure)
    with pytest.raises(AttributeError):
        success.message = b"other"
    with pytest.raises(AttributeError):
        failure.reason = "other"
