"""Tests for the composition task (Qu 2026, Sec. 5)."""

from primitives_paths.task import (
    EOS_ID, apply_op, apply_ops, full_sequence, handoff_prompt,
    handoff_response, make_problem, parse_response, prompt_tokens,
    response_tokens, verify_response,
)


def test_apply_op_f():
    assert apply_op("12345", "F") == "23451"
    assert apply_op("00000", "F") == "00000"
    assert apply_op("90123", "F") == "01239"


def test_apply_op_g():
    assert apply_op("12345", "G") == "42345"  # 1+3=4
    assert apply_op("92345", "G") == "22345"  # 9+3=12 -> 2
    assert apply_op("72345", "G") == "02345"  # 7+3=10 -> 0


def test_apply_ops_chain():
    assert apply_ops("12345", ["F", "G"]) == ["23451", "53451"]
    assert apply_ops("12345", ["G", "F"]) == ["42345", "23454"]


def test_make_problem_has_both_ops():
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(200):
        _, ops = make_problem(rng)
        assert 2 <= len(ops) <= 4
        assert "F" in ops and "G" in ops


def test_verify_accepts_correct():
    start, ops = "12345", ["F", "G", "F"]
    assert verify_response(start, ops, response_tokens(apply_ops(start, ops)))


def test_verify_rejects_wrong_intermediate():
    start, ops = "12345", ["F", "G"]
    wrong = response_tokens(["23451", "99999"])
    assert not verify_response(start, ops, wrong)


def test_verify_rejects_wrong_final():
    start, ops = "12345", ["F", "G"]
    wrong = response_tokens(["23451", "53452"])
    assert not verify_response(start, ops, wrong)


def test_verify_rejects_missing_eos():
    start, ops = "12345", ["F"]
    toks = response_tokens(apply_ops(start, ops))[:-1]
    assert not verify_response(start, ops, toks)


def test_parse_response_bad_tokens():
    assert parse_response([10, 11, EOS_ID]) is None  # op ids in response
    assert parse_response([1, 2, 3]) is None  # no EOS, short state


def test_full_sequence_roundtrip():
    start, ops = "98765", ["G", "F", "G"]
    full = full_sequence(start, ops)
    plen = len(prompt_tokens(start, ops))
    resp = full[plen:]
    assert verify_response(start, ops, resp)
    assert full[-1] == EOS_ID


def test_handoff_prompt_response():
    start, ops = "12345", ["F", "G", "F"]
    states = apply_ops(start, ops)
    # t=1: prompt ends with state1 + '='; completion is states 2,3 + EOS.
    p = handoff_prompt(start, ops, 1)
    r = handoff_response(start, ops, 1)
    assert p[-1] == 12
    assert r == response_tokens(states[1:])
    # prompt states + completion reconstruct the full correct response
    supplied = response_tokens(states[:1])[:-1]  # state1, no EOS
    assert verify_response(start, ops, supplied + [12] + r)
    # t=0 is the plain prompt
    assert handoff_prompt(start, ops, 0) == prompt_tokens(start, ops)
    assert handoff_response(start, ops, 0) == response_tokens(states)
    # t=2 leaves only the last state to complete
    assert handoff_response(start, ops, 2) == response_tokens(states[2:])
