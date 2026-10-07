from imf_ptq.imf_metrics import (
    build_record,
    format_fingerprint_prompt,
    strip_terminal_special_tokens,
    summarize_records,
)
from imf_ptq.stega.types import DecodeFailure, DecodeSuccess


def test_payload_success_is_independent_of_exact_target_match():
    record = build_record(
        "imf_000", "query", "exact target", "surface variant", 1.25,
        verifier=lambda _: DecodeSuccess(b"This is my model!"),
        expected_message=b"This is my model!",
    )
    assert record["exact_match"] is False
    assert record["decoded_payload"] == "This is my model!"
    assert record["decode_failure"] is None
    assert record["payload_match"] is True


def test_decode_failure_is_explicit_and_not_truthy_success():
    record = build_record(
        "imf_001", "query", "target", "output", 2.0,
        verifier=lambda _: DecodeFailure("truncated frame"),
        expected_message=b"This is my model!",
    )
    assert record["decoded_payload"] is None
    assert record["decode_failure"] == "truncated frame"
    assert record["payload_match"] is False


def test_summary_keeps_native_and_exact_metrics_separate():
    rows = [
        {"exact_match": False, "payload_match": True, "sequence_target_nll": 1.0, "mean_target_logprob": -1.0},
        {"exact_match": True, "payload_match": False, "sequence_target_nll": 3.0, "mean_target_logprob": -3.0},
    ]
    summary = summarize_records(rows, false_verification_rate=0.1)
    assert summary == {
        "n": 2,
        "exact_success": 1,
        "exact_rate": 0.5,
        "mean_sequence_target_nll": 2.0,
        "mean_target_logprob": -2.0,
        "payload_success": 1,
        "payload_rate": 0.5,
        "false_verification_rate": 0.1,
    }


def test_eval_prompt_exactly_matches_cau_trainer_template():
    assert format_fingerprint_prompt("system", "query") == (
        "### Instruction:\nsystem\n\n### human:\nquery\n\n### Assistant:"
    )


def test_only_boundary_special_tokens_are_removed_before_adg_decode():
    assert strip_terminal_special_tokens([1, 9, 2, 2], bos_token_id=1, eos_token_id=2, pad_token_id=2) == [9]
    assert strip_terminal_special_tokens([9, 1, 8], bos_token_id=1, eos_token_id=2, pad_token_id=0) == [9, 1, 8]
