"""SFT checkpoint construction (paper Sec. 5) and GRPO training.

Checkpoints (each SFT stage is 1 epoch unless noted):
  M1:         F-only single-op examples.
  M1+C(n):    M1, then continued SFT on n mixed-composition examples.
  M2:         F-only + G-only single-op examples, never mixed.
  M2+C(n):    M2, then continued SFT on n mixed-composition examples.

GRPO: binary reward on fully-correct trajectories (all intermediate states
and the final answer), group G=8, clip 0.2, k1 KL penalty against the
checkpoint reference (beta=0.02), one Adam step per sampled batch.
"""

import numpy as np

from .measure import MAX_NEW, _batch_generate
from .task import (
    EOS_ID, PAD_ID, apply_ops, full_sequence, make_problem, prompt_tokens,
    random_string, verify_response,
)


# ---------------------------------------------------------------- SFT data
def sft_examples(rng, kind, n_single=4000, n_comp=0, seed=None):
    """Build an SFT example list: (start, ops) pairs.

    kind in {"F", "FG", "comp"}: F-only, F+G single-op, mixed compositions.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    out = []
    if kind == "F":
        for _ in range(n_single):
            out.append((random_string(rng), ["F"]))
    elif kind == "G":
        for _ in range(n_single):
            out.append((random_string(rng), ["G"]))
    elif kind == "FG":
        for _ in range(n_single):
            out.append((random_string(rng), ["F"]))
        for _ in range(n_single):
            out.append((random_string(rng), ["G"]))
    elif kind == "comp":
        for _ in range(n_comp):
            out.append(make_problem(rng))
    else:
        raise ValueError(kind)
    return out


def _batch_arrays(examples, idxs):
    """Pad a batch of (start, ops) examples; return tokens, mask.

    mask marks response positions in the full sequence (1 = response token).
    """
    seqs = []
    masks = []
    for i in idxs:
        start, ops = examples[i]
        full = full_sequence(start, ops)
        plen = len(prompt_tokens(start, ops))
        m = [0] * plen + [1] * (len(full) - plen)
        seqs.append(full)
        masks.append(m)
    width = max(len(s) for s in seqs)
    tokens = np.full((len(seqs), width), PAD_ID, dtype=int)
    mask = np.zeros((len(seqs), width), dtype=float)
    for j, (s, m) in enumerate(zip(seqs, masks)):
        tokens[j, :len(s)] = s
        mask[j, :len(m)] = m
    return tokens, mask


def sft_train(model, examples, epochs=1, batch=64, lr=3e-3, rng=None,
              log_every=0):
    """Adam SFT on (start, ops) examples. Returns list of mean batch losses."""
    rng = rng or np.random.default_rng(0)
    n = len(examples)
    losses = []
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idxs = order[s:s + batch]
            tokens, mask = _batch_arrays(examples, idxs)
            loss, grads, _ = model.loss_and_grads(tokens, mask)
            model.adam_step(grads, lr)
            losses.append(float(loss))
            if log_every and len(losses) % log_every == 0:
                print(f"  sft step {len(losses)} loss {loss:.4f}", flush=True)
    return losses


# ---------------------------------------------------------------- sampling
def sample_trajectories(model, problems, group, temperature, rng):
    """Sample `group` trajectories per problem at temperature.

    Returns (full_seqs, rewards, resp_lens): full token lists (prompt +
    sampled response, truncated at EOS), binary rewards, and response
    lengths in tokens (incl. EOS).
    """
    prompts = [prompt_tokens(s, ops) for s, ops in problems]
    reps = []
    for p in prompts:
        reps += [p] * group
    gens = _batch_generate(model, reps, MAX_NEW, temperature, False, rng)
    full_seqs, rewards, resp_lens = [], [], []
    k = 0
    for pi, (s, ops) in enumerate(problems):
        for _ in range(group):
            g = gens[k]
            k += 1
            full_seqs.append(prompts[pi] + g)
            rewards.append(1.0 if verify_response(s, ops, g) else 0.0)
            resp_lens.append(len(g))
    return full_seqs, np.array(rewards), resp_lens


def _padded_logp_inputs(full_seqs, prompt_lens):
    """Pad full sequences; return tokens, response mask (shifted frame).

    resp_mask[b, j] = 1 iff tokens[b, j+1] is a response token.
    """
    width = max(len(s) for s in full_seqs)
    tokens = np.full((len(full_seqs), width), PAD_ID, dtype=int)
    resp = np.zeros((len(full_seqs), width - 1), dtype=float)
    for b, seq in enumerate(full_seqs):
        tokens[b, :len(seq)] = seq
        plen = prompt_lens[b]
        # response tokens are at full positions plen..len(seq)-1;
        # in the shifted frame they are indices plen-1..len(seq)-2.
        resp[b, plen - 1:len(seq) - 1] = 1.0
    return tokens, resp


def grpo_update(model, ref_model, problems, group=8, lr=1e-3, clip=0.2,
                beta=0.02, temperature=1.0, rng=None):
    """One GRPO update on a batch of problems. Returns a stats dict."""
    rng = rng or np.random.default_rng(0)
    full_seqs, rewards, _ = sample_trajectories(model, problems, group,
                                                temperature, rng)
    B = len(problems)
    prompt_lens = []
    for s, ops in problems:
        prompt_lens += [len(prompt_tokens(s, ops))] * group
    tokens, resp = _padded_logp_inputs(full_seqs, prompt_lens)

    # logp under the current policy (before the step) and the reference.
    logp_old = model.logprobs(tokens)
    logp_ref = ref_model.logprobs(tokens)
    # per-trajectory mean logp over response tokens
    n_tok = resp.sum(axis=1, keepdims=True)
    mean_lp = (logp_old * resp).sum(axis=1) / np.maximum(n_tok[:, 0], 1)
    mean_ref = (logp_ref * resp).sum(axis=1) / np.maximum(n_tok[:, 0], 1)

    # group-normalised advantages (per problem, over its `group` samples)
    r = rewards.reshape(B, group)
    adv = np.zeros_like(r)
    for i in range(B):
        mu = r[i].mean()
        sd = r[i].std()
        if sd > 1e-8:
            adv[i] = (r[i] - mu) / sd
    adv_flat = adv.reshape(-1)
    signal = float(np.mean([1.0 if r[i].std() > 1e-8 else 0.0 for i in range(B)]))
    if signal == 0.0:
        # No group has mixed rewards, so every advantage is zero and the
        # update has no learning signal. Skip the step entirely: the KL
        # term alone would otherwise drift the policy away from the
        # reference (Adam takes a full-size step in the direction of the
        # tiny KL gradient), destroying primitives for nothing.
        return {
            "mean_reward": float(rewards.mean()),
            "signal_frac": 0.0,
            "loss": 0.0,
            "kl": 0.0,
            "mean_ratio": 1.0,
            "skipped": True,
        }

    # PPO-style clipped surrogate + k1 KL penalty, per response token.
    logits, cache = model.forward(tokens)
    pred = logits[:, :-1, :]
    mx = pred.max(axis=-1, keepdims=True)
    e = np.exp(pred - mx)
    probs = e / e.sum(axis=-1, keepdims=True)
    tgt = tokens[:, 1:]
    lp_new = np.log(probs[np.arange(len(tokens))[:, None],
                          np.arange(tokens.shape[1] - 1)[None, :], tgt] + 1e-12)
    ratio = np.exp(lp_new - logp_old)
    clipped = np.clip(ratio, 1 - clip, 1 + clip)
    a_tok = np.repeat(adv_flat, 1)[:, None] * np.ones_like(lp_new)
    # per-token: unclipped branch active where ratio*A <= clip*A
    unclipped_active = (ratio * a_tok) <= (clipped * a_tok) + 1e-12
    w = np.where(unclipped_active, a_tok * ratio, 0.0)
    # loss = -mean(w * lp_new) + beta * mean(lp_new - logp_ref) over resp toks
    n_resp = resp.sum()
    loss_pg = -(w * lp_new * resp).sum() / max(n_resp, 1)
    kl = ((lp_new - logp_ref) * resp).sum() / max(n_resp, 1)
    loss = loss_pg + beta * kl

    # gradient: d loss / d logits = (w - beta) * (probs - onehot) / n_resp
    # on response tokens (w is 0 on clipped tokens; KL term always active).
    # Sign check: d logp / d logits = onehot - probs = -(probs - onehot),
    # so the -w*lp term contributes +w*(probs - onehot) (descending the
    # loss raises the logprob of positive-advantage trajectories) and the
    # +beta*KL term contributes -beta*(probs - onehot) (pulling the policy
    # back toward the reference). The original code had (w + beta), which
    # flipped the KL term and made the policy drift away from the
    # reference even with zero reward signal.
    d = probs.copy()
    d[np.arange(len(tokens))[:, None], np.arange(tokens.shape[1] - 1)[None, :], tgt] -= 1.0
    d *= ((w - beta) * resp / max(n_resp, 1))[:, :, None]
    dlogits = np.zeros_like(logits)
    dlogits[:, :-1, :] = d
    grads = model._backward(dlogits, cache)
    model.adam_step(grads, lr)
    return {
        "mean_reward": float(rewards.mean()),
        "signal_frac": signal,
        "loss": float(loss),
        "kl": float(kl),
        "mean_ratio": float((ratio * resp).sum() / max(n_resp, 1)),
        "skipped": False,
    }


def grpo_train(model, ref_model, train_problems, updates=150, batch=16,
               group=8, lr=1e-3, clip=0.2, beta=0.02, temperature=1.0,
               rng=None, log_every=25):
    """Run GRPO for `updates` steps. Returns list of per-update stats."""
    rng = rng or np.random.default_rng(0)
    model.reset_optimizer()  # fresh optimizer for the RL phase
    stats = []
    for u in range(updates):
        idx = rng.integers(0, len(train_problems), size=batch)
        problems = [train_problems[i] for i in idx]
        st = grpo_update(model, ref_model, problems, group=group, lr=lr,
                         clip=clip, beta=beta, temperature=temperature, rng=rng)
        st["update"] = u
        stats.append(st)
        if log_every and (u + 1) % log_every == 0:
            print(f"  grpo {u+1}/{updates} reward {st['mean_reward']:.3f} "
                  f"signal {st['signal_frac']:.2f} kl {st['kl']:.4f}",
                  flush=True)
    return stats
