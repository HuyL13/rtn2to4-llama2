import random

import pytest

from imf_ptq.stega import DecodeFailure, DecodeSuccess
from imf_ptq.stega.adg import ADGCodec, adg_group


KEY = bytes(range(32))
MESSAGE = b"This is my model!"


class UniformCarrier:
    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size

    def distribution(self, prefix: list[int]) -> list[float]:
        return [1.0] * self.vocabulary_size


class InvalidCarrier:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def distribution(self, prefix: list[int]) -> list[float]:
        return self.probabilities


GROUPED_PROBABILITIES = [0.24, 0.21, 0.17, 0.13, 0.09, 0.07, 0.05, 0.04]


class GroupedCarrier:
    def distribution(self, prefix: list[int]) -> list[float]:
        return GROUPED_PROBABILITIES


class ContinuationFilteredCarrier:
    def distribution(self, prefix: list[int]) -> list[float]:
        return [1.0] * 256

    def is_stable_continuation(self, prefix, token_id):
        return token_id != 0


class PrefixSensitiveCarrier:
    def distribution(self, prefix: list[int]) -> list[float]:
        if prefix and prefix[0] == 0:
            return list(reversed(GROUPED_PROBABILITIES))
        return GROUPED_PROBABILITIES


def test_grouping_assigns_each_remaining_token_to_nearest_residual_group():
    # Parallel least-mass assignment produces a different partition.
    assert adg_group(GROUPED_PROBABILITIES) == [[0], [1, 7], [2, 4], [3, 5, 6]]


def test_grouping_breaks_probability_ties_by_token_id():
    assert adg_group([0.25, 0.25, 0.25, 0.25]) == [[0], [1], [2], [3]]


def test_encode_is_deterministic_for_same_seed_key_and_carrier():
    codec = ADGCodec(UniformCarrier(256), max_tokens=64, seed=42)

    assert codec.encode(MESSAGE, KEY) == codec.encode(MESSAGE, KEY)


def test_non_singleton_group_sampling_is_deterministic_and_probability_weighted():
    encoded = [
        ADGCodec(GroupedCarrier(), max_tokens=200, seed=seed).encode(b"", KEY)[0]
        for seed in range(64)
    ]

    assert set(encoded) <= {2, 4}
    assert set(encoded) == {2, 4}
    assert encoded.count(2) > encoded.count(4)
    assert encoded == [
        ADGCodec(GroupedCarrier(), max_tokens=200, seed=seed).encode(b"", KEY)[0]
        for seed in range(64)
    ]


def test_encode_samples_only_stable_continuations_when_carrier_provides_filter():
    codec = ADGCodec(ContinuationFilteredCarrier(), max_tokens=64, seed=42)

    assert 0 not in codec.encode(MESSAGE, KEY)


def test_adg_round_trip_recovers_authenticated_message():
    codec = ADGCodec(UniformCarrier(256), max_tokens=64, seed=42)

    decoded = codec.decode_tokens(codec.encode(MESSAGE, KEY), KEY)

    assert decoded == DecodeSuccess(MESSAGE)


def test_wrong_key_fails_closed():
    codec = ADGCodec(UniformCarrier(256), max_tokens=64, seed=42)

    decoded = codec.decode_tokens(codec.encode(MESSAGE, KEY), bytes(reversed(KEY)))

    assert isinstance(decoded, DecodeFailure)


def test_truncated_carrier_fails_closed():
    codec = ADGCodec(UniformCarrier(256), max_tokens=64, seed=42)
    tokens = codec.encode(MESSAGE, KEY)

    decoded = codec.decode_tokens(tokens[:-1], KEY)

    assert isinstance(decoded, DecodeFailure)


def test_altered_token_fails_closed():
    codec = ADGCodec(UniformCarrier(256), max_tokens=64, seed=42)
    tokens = codec.encode(MESSAGE, KEY)
    tokens[5] ^= 1

    decoded = codec.decode_tokens(tokens, KEY)

    assert isinstance(decoded, DecodeFailure)


def test_cross_group_token_change_corrupts_the_following_prefix_and_fails_closed():
    codec = ADGCodec(PrefixSensitiveCarrier(), max_tokens=200, seed=42)
    tokens = codec.encode(MESSAGE, KEY)
    tokens[0] = 0

    assert isinstance(codec.decode_tokens(tokens, KEY), DecodeFailure)


def test_appended_valid_token_is_rejected_after_complete_frame():
    codec = ADGCodec(GroupedCarrier(), max_tokens=200, seed=42)
    tokens = codec.encode(MESSAGE, KEY)

    assert isinstance(codec.decode_tokens(tokens + [1], KEY), DecodeFailure)


@pytest.mark.parametrize("probabilities", [[float("nan"), 1.0], [float("inf"), 1.0], [-0.1, 1.1], [0.0, 0.0]])
def test_invalid_probability_mass_is_rejected(probabilities: list[float]):
    codec = ADGCodec(InvalidCarrier(probabilities), max_tokens=64, seed=42)

    with pytest.raises(ValueError, match="probabil"):
        codec.encode(MESSAGE, KEY)
    assert isinstance(codec.decode_tokens([0], KEY), DecodeFailure)


def test_zero_capacity_distribution_cannot_encode():
    codec = ADGCodec(InvalidCarrier([0.9, 0.1]), max_tokens=64, seed=42)

    with pytest.raises(ValueError, match="capacity"):
        codec.encode(MESSAGE, KEY)


def test_insufficient_token_budget_cannot_encode_complete_frame():
    codec = ADGCodec(UniformCarrier(256), max_tokens=37, seed=42)

    with pytest.raises(ValueError, match="max_tokens"):
        codec.encode(MESSAGE, KEY)


def test_encoding_does_not_mutate_global_random_state():
    codec = ADGCodec(UniformCarrier(256), max_tokens=64, seed=42)
    random.seed(1729)
    expected_after_encode = random.Random(1729)
    expected_after_encode.random()

    random.random()
    codec.encode(MESSAGE, KEY)

    assert random.random() == expected_after_encode.random()
