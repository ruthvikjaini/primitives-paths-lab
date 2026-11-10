"""Pre-RL measurement protocol (Qu 2026, Sec. 4) and evaluation helpers.

- Effective full-solution support: pass@k via the unbiased estimator of
  Chen et al. (2021), n=64 samples per problem, temperature 0.7.
- Atomic primitive coverage: greedy accuracy on isolated single-op probes;
  coverage = min over primitives.
- Handoff curve H(t): sample completions given the first t reference
  states; composition gap G_comp(t) = H(t) - H(0).
- Training-prompt support: fraction of RL training prompts solved at least
  once (the Gemma-M2 anomaly check).
"""

import math

import numpy as np

from .task import (
    EOS_ID, PAD_ID, apply_ops, handoff_prompt, handoff_response,
    make_problem, prompt_tokens, single_op_prompt, verify_response,
)

MAX_NEW = 26


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if c == 0:
        return 0.0
    if k > n - c:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _batch_generate(model, prompt_list, max_new, temperature, greedy, rng,
                    chunk=512):
    """Generate completions for a list of prompts, in chunks.

    Prompts are grouped by length internally: generation must start right
    after the last real prompt token (no padding between prompt and output).
    Returns a list of token lists (one per prompt), truncated at EOS.
    """
    assert max(len(p) for p in prompt_list) + max_new <= 40, (
        "prompt + max_new exceeds the 40-position table; reduce max_new")
    out = [None] * len(prompt_list)
    groups = _grouped(prompt_list)
    for length, idxs in groups.items():
        for s in range(0, len(idxs), chunk):
            sub = idxs[s:s + chunk]
            arr = np.array([prompt_list[i] for i in sub], dtype=int)
            gen = model.generate(arr, max_new, temperature=temperature,
                                 greedy=greedy, rng=rng)
            for j, row in zip(sub, gen):
                toks = []
                for t in row:
                    if t == EOS_ID:
                        toks.append(int(t))
                        break
                    toks.append(int(t))
                out[j] = toks
    return out


def _grouped(prompts):
    """Group prompt indices by prompt length (generation needs equal lengths)."""
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    return groups


def sample_successes(model, problems, n_samples, temperature, rng):
    """For each problem, count successes among n_samples trajectories.

    Returns a list of counts (one per problem).
    """
    counts = [0] * len(problems)
    prompts = [prompt_tokens(s, ops) for s, ops in problems]
    for length, idxs in _grouped(prompts).items():
        reps = []
        for i in idxs:
            reps += [prompts[i]] * n_samples
        gens = _batch_generate(model, reps, MAX_NEW, temperature, False, rng)
        k = 0
        for i in idxs:
            start, ops = problems[i]
            c = 0
            for _ in range(n_samples):
                if verify_response(start, ops, gens[k]):
                    c += 1
                k += 1
            counts[i] = c
    return counts


def evaluate_pass(model, problems, n_samples=64, temperature=0.7, rng=None,
                  ks=(1, 2, 4, 8, 16, 32, 64)):
    """pass@k table + per-problem success counts."""
    counts = sample_successes(model, problems, n_samples, temperature, rng)
    table = {}
    for k in ks:
        table[k] = float(np.mean([pass_at_k(n_samples, c, k) for c in counts]))
    return {"pass": table, "counts": counts,
            "mean_success_rate": float(np.mean(counts)) / n_samples}


def probe_primitives(model, rng, n=200):
    """Greedy isolated single-op execution accuracy for F and G."""
    accs = {}
    for op in ("F", "G"):
        correct = 0
        prompts = []
        starts = []
        for _ in range(n):
            s = "".join(rng.choice(list("0123456789")) for _ in range(5))
            starts.append(s)
            prompts.append(single_op_prompt(s, op))
        gens = _batch_generate(model, prompts, 8, 1.0, True, rng)
        for s, g in zip(starts, gens):
            if verify_response(s, [op], g):
                correct += 1
        accs[op] = correct / n
    accs["coverage"] = min(accs["F"], accs["G"])
    return accs


def handoff_curve(model, problems, n_samples=16, temperature=0.7, rng=None):
    """H(t) for t = 0, 1, 2 on problems with exactly 3 ops.

    H(t) is the fraction of sampled completions that correctly finish the
    trajectory given the first t reference states.
    """
    probs = [p for p in problems if len(p[1]) == 3]
    curve = {}
    for t in (0, 1, 2):
        # remaining response is (3 - t) states: 6 tokens each + margin
        max_new = (3 - t) * 6 + 2
        prompts = [handoff_prompt(s, ops, t) for s, ops in probs]
        reps = []
        for p in prompts:
            reps += [p] * n_samples
        gens = _batch_generate(model, reps, max_new, temperature, False, rng)
        ok = 0
        total = 0
        k = 0
        for s, ops in probs:
            want = handoff_response(s, ops, t)
            for _ in range(n_samples):
                # generated tokens must equal the remaining states + EOS
                if gens[k] == want:
                    ok += 1
                total += 1
                k += 1
        curve[t] = ok / max(total, 1)
    curve["gap1"] = curve[1] - curve[0]
    curve["gap2"] = curve[2] - curve[0]
    return curve


def training_support(model, problems, n_samples=64, temperature=0.7, rng=None):
    """Fraction of training prompts solved at least once in n samples."""
    counts = sample_successes(model, problems, n_samples, temperature, rng)
    solved = sum(1 for c in counts if c > 0)
    return {"n_prompts": len(problems), "n_solved": solved,
            "frac_solved": solved / max(len(problems), 1),
            "mean_c": float(np.mean(counts)) if counts else 0.0}
