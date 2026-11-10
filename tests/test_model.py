"""Tests for the tiny transformer: gradient check, loss, generation."""

import numpy as np

from primitives_paths.model import Transformer
from primitives_paths.task import VOCAB_SIZE, full_sequence


def _small_batch():
    # two short single-op examples
    seqs = [full_sequence("12345", ["F"]), full_sequence("98765", ["G"])]
    width = max(len(s) for s in seqs)
    tokens = np.full((2, width), 14, dtype=int)
    mask = np.zeros((2, width), dtype=float)
    for j, s in enumerate(seqs):
        tokens[j, :len(s)] = s
        # prompt is 5 digits + 1 op + '=' = 7 tokens; response starts at 7
        mask[j, 7:] = 1.0
    return tokens, mask


def test_loss_decreases_with_sft_steps():
    rng = np.random.default_rng(0)
    model = Transformer(VOCAB_SIZE, seed=1)
    tokens, mask = _small_batch()
    l0, _, _ = model.loss_and_grads(tokens, mask)
    for _ in range(30):
        loss, grads, _ = model.loss_and_grads(tokens, mask)
        model.adam_step(grads, 3e-3)
    l1, _, _ = model.loss_and_grads(tokens, mask)
    assert l1 < l0 * 0.5, (l0, l1)


def test_gradient_check_subset():
    """Finite-difference check on a random subset of parameters."""
    rng = np.random.default_rng(7)
    model = Transformer(VOCAB_SIZE, seed=3)
    tokens, mask = _small_batch()

    def loss_fn():
        l, _, _ = model.loss_and_grads(tokens, mask)
        return l

    _, grads, _ = model.loss_and_grads(tokens, mask)
    keys = ["Wte", "Wq", "Wk", "Wv", "Wo", "W1", "W2", "Wh",
            "ln1_g", "ln2_g", "lnf_g", "bq", "b1", "bh"]
    eps = 1e-5
    worst = 0.0
    for key in keys:
        p = model.params[key]
        idx = [tuple(rng.integers(0, s) for s in p.shape) for _ in range(3)]
        for ix in idx:
            old = p[ix]
            p[ix] = old + eps
            lp = loss_fn()
            p[ix] = old - eps
            lm = loss_fn()
            p[ix] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[key][ix]
            denom = max(abs(num), abs(ana), 1e-8)
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    assert worst < 1e-4, f"worst relative gradient error {worst}"


def test_logprobs_matches_forward():
    model = Transformer(VOCAB_SIZE, seed=5)
    tokens, _ = _small_batch()
    lp = model.logprobs(tokens)
    assert lp.shape == (2, tokens.shape[1] - 1)
    # log-probs of the true next tokens should be finite and <= 0
    assert np.all(np.isfinite(lp))
    assert np.all(lp <= 1e-6)


def test_generate_greedy_format():
    rng = np.random.default_rng(0)
    model = Transformer(VOCAB_SIZE, seed=2)
    prompt = np.array([[1, 2, 3, 4, 5, 10, 12]], dtype=int)
    out = model.generate(prompt, 26, greedy=True)
    assert out.shape == (1, 26)
    assert np.all((out >= 0) & (out < VOCAB_SIZE))
    # after the first EOS, rows are PAD (14)
    row = out[0]
    if 13 in row:
        i = int(np.where(row == 13)[0][0])
        assert np.all(row[i + 1:] == 14)


def test_generate_sampling_shape():
    rng = np.random.default_rng(0)
    model = Transformer(VOCAB_SIZE, seed=2)
    prompt = np.array([[1, 2, 3, 4, 5, 10, 12]] * 4, dtype=int)
    out = model.generate(prompt, 10, temperature=0.7, rng=rng)
    assert out.shape == (4, 10)
