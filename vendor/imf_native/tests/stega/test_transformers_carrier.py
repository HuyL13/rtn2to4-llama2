from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from imf_ptq.stega.transformers_carrier import TransformersCarrier


class FakeTokenizer:
    def __init__(self, vocabulary_size: int) -> None:
        self.vocab_size = vocabulary_size

    def __len__(self) -> int:
        return self.vocab_size


class AddedTokenTokenizer:
    vocab_size = 3

    def __len__(self) -> int:
        return 4


class RoundtripTokenizer:
    vocab_size = 4
    all_special_ids = [0]

    def __len__(self) -> int:
        return 4

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return {0: "<s>", 1: " stable", 2: "", 3: " split"}[ids[0]]

    def encode(self, text, add_special_tokens=False):
        return {" stable": [1], " split": [1, 3]}.get(text, [])


class FakeModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.input_ids: torch.Tensor | None = None

    def __call__(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        self.input_ids = input_ids.clone()
        return SimpleNamespace(logits=self.logits)


def test_distribution_uses_float32_stable_softmax_probabilities():
    # Removing float32 conversion or stable softmax must break this overflow-prone case.
    model = FakeModel(torch.tensor([[[1000.0, 1001.0, 999.0]]], dtype=torch.float64))
    carrier = TransformersCarrier(model, FakeTokenizer(3), prefix_ids=[10])

    probabilities = carrier.distribution([])

    assert probabilities == pytest.approx([0.24472847, 0.66524096, 0.09003057])
    assert sum(probabilities) == pytest.approx(1.0)


def test_distribution_concatenates_fixed_and_generation_prefixes():
    model = FakeModel(torch.tensor([[[0.0, 0.0, 0.0]]]))
    carrier = TransformersCarrier(model, FakeTokenizer(3), prefix_ids=[10, 11])

    carrier.distribution([12, 13])

    assert model.input_ids is not None
    assert model.input_ids.tolist() == [[10, 11, 12, 13]]


def test_distribution_prefers_total_tokenizer_length_for_added_tokens():
    model = FakeModel(torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]))
    carrier = TransformersCarrier(model, AddedTokenTokenizer(), prefix_ids=[10])

    probabilities = carrier.distribution([])

    assert len(probabilities) == 4


def test_distribution_masks_unstable_and_special_tokens():
    model = FakeModel(torch.tensor([[[10.0, 1.0, 10.0, 10.0]]]))
    carrier = TransformersCarrier(model, RoundtripTokenizer(), prefix_ids=[10])

    probabilities = carrier.distribution([])

    assert probabilities == pytest.approx([0.0, 1.0, 0.0, 0.0])


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_carrier_rejects_invalid_temperature(temperature: float):
    with pytest.raises(ValueError, match="temperature"):
        TransformersCarrier(FakeModel(torch.zeros((1, 1, 3))), FakeTokenizer(3), [1], temperature)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros((1, 3)),
        torch.zeros((2, 1, 3)),
        torch.tensor([[[float("nan"), 0.0, 0.0]]]),
        torch.zeros((1, 1, 4)),
    ],
)
def test_distribution_rejects_invalid_model_logits(logits: torch.Tensor):
    carrier = TransformersCarrier(FakeModel(logits), FakeTokenizer(3), [1])

    with pytest.raises(ValueError, match="logits|vocabulary"):
        carrier.distribution([])
