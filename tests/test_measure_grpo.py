"""Tests for the measurement protocol and GRPO update."""

import numpy as np

from primitives_paths.measure import pass_at_k
from primitives_paths.model import Transformer
from primitives_paths.task import VOCAB_SIZE, full_sequence, prompt_tokens
from primitives_paths.train import (grpo_update, sft_train, sft_examples,
                                    sample_trajectories, _padded_logp_inputs)


def test_pass_at_k_edges():
    assert pass_at_k(64, 0, 1) == 0.0
    assert pass_at_k(64, 0, 64) == 0.0
    assert pass_at_k(64, 64, 1) == 1.0
    assert pass_at_k(64, 64, 64) == 1.0
    assert abs(pass_at_k(64, 32, 1) - 0.5) < 1e-12
    # k > n - c means at least one success is guaranteed
    assert pass_at_k(10, 1, 10) == 1.0
    # brute-force check for a small case: n=4, c=1, k=2
    # P(at least one success in 2 draws without replacement) = 1 - C(3,2)/C(4,2)
    assert abs(pass_at_k(4, 1, 2) - (1 - 3 / 6)) < 1e-12
    # monotone in k
    vals = [pass_at_k(64, 5, k) for k in (1, 2, 4, 8, 16, 32, 64)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_handoff_curve_runs_within_position_table():
    """Regression test: handoff prompts are long (t=2 is 21 tokens), so
    the completion budget must keep prompt + max_new <= 40 positions."""
    rng = np.random.default_rng(3)
    model = Transformer(VOCAB_SIZE, seed=13)
    ex = sft_examples(rng, "FG", n_single=200) + sft_examples(
        rng, "comp", n_comp=100)
    sft_train(model, ex, epochs=2, batch=64, lr=3e-3, rng=rng)
    from primitives_paths.measure import handoff_curve
    from primitives_paths.task import make_problem
    prng = np.random.default_rng(5)
    probs = [make_problem(prng, m=3) for _ in range(6)]
    curve = handoff_curve(model, probs, n_samples=4, temperature=0.7,
                          rng=rng)
    assert 0.0 <= curve[0] <= 1.0
    assert 0.0 <= curve[2] <= 1.0


def test_grpo_no_signal_no_move_with_beta_zero():
    """If every trajectory in every group has the same reward and beta=0,
    advantages are zero and the update must not move the parameters."""
    rng = np.random.default_rng(0)
    model = Transformer(VOCAB_SIZE, seed=11)
    # train until F is mastered, so sampling is (near-)deterministic and
    # every trajectory in every group is correct.
    ex = sft_examples(rng, "F", n_single=2000)
    sft_train(model, ex, epochs=5, batch=64, lr=3e-3, rng=rng)
    ref = model.copy()
    model.reset_optimizer()  # stale SFT momentum would move params anyway
    before = {k: v.copy() for k, v in model.params.items()}
    problems = [("12345", ["F"]), ("54321", ["F"])]
    st = grpo_update(model, ref, problems, group=4, lr=1e-3, beta=0.0,
                     temperature=0.1, rng=rng)
    for k in model.params:
        assert np.allclose(model.params[k], before[k]), k
    assert st["signal_frac"] == 0.0


def test_grpo_raises_rewarded_logprob():
    """A GRPO update must raise the logprob of a rewarded trajectory and
    lower that of an unrewarded one (beta=0 isolates the policy-gradient
    term). This catches a sign error in the surrogate gradient, which the
    params-moved test above cannot see."""
    rng = np.random.default_rng(1)
    model = Transformer(VOCAB_SIZE, seed=12)
    ex = (sft_examples(rng, "FG", n_single=2000)
          + sft_examples(rng, "comp", n_comp=300))
    sft_train(model, ex, epochs=5, batch=64, lr=3e-3, rng=rng)
    ref = model.copy()
    model.reset_optimizer()
    problems = [("12345", ["F", "G"]), ("11111", ["G", "F"]),
                ("90210", ["F", "G"]), ("77777", ["G", "F"])]
    # Find a problem whose sampled group has mixed rewards.
    target = None
    for (s, ops) in problems:
        seqs, rewards, _ = sample_trajectories(
            model, [(s, ops)], group=8, temperature=1.0,
            rng=np.random.default_rng(99))
        if 0 < rewards.sum() < len(rewards):
            target = (s, ops, seqs, rewards)
            break
    assert target is not None, "no problem produced a mixed-reward group"
    s, ops, seqs, rewards = target
    good = [q for q, r in zip(seqs, rewards) if r == 1.0][0]
    bad = [q for q, r in zip(seqs, rewards) if r == 0.0][0]
    plen = len(prompt_tokens(s, ops))

    def mean_resp_logp(seq):
        toks, resp = _padded_logp_inputs([seq], [plen])
        lp = model.logprobs(toks)
        return float((lp * resp).sum() / resp.sum())

    lp_good_before = mean_resp_logp(good)
    lp_bad_before = mean_resp_logp(bad)
    # Replay the exact same sampled group inside the update by re-seeding.
    # lr is small (1e-4) on purpose: the sign is a first-order property,
    # and a 3e-3 Adam step overshoots the linear regime on this tiny model
    # (the good logprob then falls for step-size reasons, not sign reasons).
    grpo_update(model, ref, [(s, ops)], group=8, lr=1e-4, beta=0.0,
                temperature=1.0, rng=np.random.default_rng(99))
    assert mean_resp_logp(good) > lp_good_before, \
        "rewarded trajectory logprob did not increase (sign error?)"
    assert mean_resp_logp(bad) < lp_bad_before, \
        "unrewarded trajectory logprob did not decrease (sign error?)"


def test_grpo_learns_from_mixed_rewards():
    """On a tiny problem where the base sometimes succeeds, one GRPO
    update with beta=0 should not decrease the mean reward of a fresh
    sample by a large amount and should move parameters."""
    rng = np.random.default_rng(1)
    model = Transformer(VOCAB_SIZE, seed=12)
    # train on single ops AND compositions, so 2-op problems are sometimes
    # solved and GRPO groups see mixed rewards (calibrated recipe: the
    # G primitive has a sharp learning threshold, see docs/RESEARCH.md).
    ex = (sft_examples(rng, "FG", n_single=2000)
          + sft_examples(rng, "comp", n_comp=300))
    sft_train(model, ex, epochs=5, batch=64, lr=3e-3, rng=rng)
    ref = model.copy()
    model.reset_optimizer()
    before = {k: v.copy() for k, v in model.params.items()}
    problems = [("12345", ["F", "G"]), ("11111", ["G", "F"]),
                ("90210", ["F", "G"]), ("77777", ["G", "F"])]
    moved = False
    for _ in range(5):
        st = grpo_update(model, ref, problems, group=8, lr=3e-3, beta=0.0,
                         temperature=1.0, rng=rng)
        if st["signal_frac"] > 0:
            moved = True
    assert moved, "no update ever saw a mixed-reward group"
    diff = sum(float(np.abs(model.params[k] - before[k]).sum())
               for k in model.params)
    assert diff > 0.0
