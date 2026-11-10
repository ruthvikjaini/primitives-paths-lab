"""Extensions (paper Sec. 6 limitations as prompts).

E2 — In-context primitive probe. The paper's M1+C300 checkpoints retain
     near-zero isolated-G accuracy while using G inside complete
     trajectories. Test whether the probe, not the skill, is missing:
     compare isolated G/F accuracy with contextual accuracy — the
     handoff t=1 completion on 2-op problems, where the only remaining
     step is one primitive applied to a supplied state.
E3 — Prospective-prediction test + threshold sensitivity. Spearman rank
     correlation between pre-RL coordinates (support pass@64, coverage)
     and post-RL gain (Delta pass@1) across the six checkpoints; regime
     classification over a grid of (tau_p, tau_c) thresholds.
"""

import argparse
import json
import os

import numpy as np

from primitives_paths.measure import _batch_generate
from primitives_paths.model import Transformer
from primitives_paths.task import VOCAB_SIZE, handoff_prompt, handoff_response, make_problem


def contextual_probe(model, rng, n_per_order=100, n_samples=16,
                     temperature=0.7):
    """Contextual F/G accuracy from 2-op handoff t=1 completions.

    [F, G] problems: completion applies G to the supplied state.
    [G, F] problems: completion applies F to the supplied state.
    """
    out = {}
    for ops, prim in ((["F", "G"], "G"), (["G", "F"], "F")):
        probs = []
        for _ in range(n_per_order):
            s = "".join(rng.choice(list("0123456789")) for _ in range(5))
            probs.append((s, ops))
        prompts = [handoff_prompt(s, ops, 1) for s, ops in probs]
        reps = []
        for p in prompts:
            reps += [p] * n_samples
        # remaining response is one state + EOS: 6 tokens (+margin)
        gens = _batch_generate(model, reps, 8, temperature, False, rng)
        ok = 0
        total = 0
        k = 0
        for s, o in probs:
            want = handoff_response(s, o, 1)
            for _ in range(n_samples):
                if gens[k] == want:
                    ok += 1
                total += 1
                k += 1
        out[prim] = ok / max(total, 1)
    out["coverage_contextual"] = min(out["F"], out["G"])
    return out


def spearman(x, y):
    """Spearman rank correlation (no scipy dependency)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ranks for ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        for val, c in zip(np.unique(v), counts):
            if c > 1:
                r[v == val] = r[v == val].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def classify(p0, c, tau_p, tau_c):
    if p0 >= tau_p and c >= tau_c:
        return "amplification"
    if p0 < tau_p and c >= tau_c:
        return "composition-limited"
    if p0 < tau_p and c < tau_c:
        return "acquisition-limited"
    return "diagnostic-mismatch"


def main(ckpt_dir, pre_path, grpo_summary_path, out_path):
    with open(pre_path) as f:
        pre = json.load(f)
    with open(grpo_summary_path) as f:
        summ = json.load(f)
    ckpts = ["M1", "M1+C100", "M1+C300", "M2", "M2+C100", "M2+C300"]

    # E2: contextual probes (pre-RL checkpoints)
    rng = np.random.default_rng(777)
    e2 = {}
    for name in ckpts:
        model = Transformer.load(os.path.join(ckpt_dir, f"{name}.npz"),
                                 VOCAB_SIZE)
        e2[name] = contextual_probe(model, rng)
        print(f"E2 {name}: isolated F/G "
              f"{pre['checkpoints'][name]['probes']['F']:.2f}/"
              f"{pre['checkpoints'][name]['probes']['G']:.2f} vs contextual "
              f"{e2[name]['F']:.2f}/{e2[name]['G']:.2f}", flush=True)

    # E3: post-RL gains (seed means) and rank correlations
    gains = {}
    for name in ckpts:
        posts1 = [summ[f"{name}_seed{s}"]["post_pass"]["1"] for s in (0, 1, 2)
                  if f"{name}_seed{s}" in summ]
        posts64 = [summ[f"{name}_seed{s}"]["post_pass"]["64"] for s in (0, 1, 2)
                   if f"{name}_seed{s}" in summ]
        pre1 = pre["checkpoints"][name]["pass"]["1"]
        pre64 = pre["checkpoints"][name]["pass"]["64"]
        gains[name] = {
            "post_pass1_mean": float(np.mean(posts1)),
            "post_pass64_mean": float(np.mean(posts64)),
            "delta_pass1": float(np.mean(posts1)) - pre1,
            "delta_pass64": float(np.mean(posts64)) - pre64,
        }
    support = [pre["checkpoints"][n]["pass"]["64"] for n in ckpts]
    coverage = [pre["checkpoints"][n]["probes"]["coverage"] for n in ckpts]
    d1 = [gains[n]["delta_pass1"] for n in ckpts]
    d64 = [gains[n]["delta_pass64"] for n in ckpts]
    e3 = {
        "gains": gains,
        "spearman_support_vs_delta_pass1": spearman(support, d1),
        "spearman_support_vs_delta_pass64": spearman(support, d64),
        "spearman_coverage_vs_delta_pass1": spearman(coverage, d1),
        "spearman_coverage_vs_delta_pass64": spearman(coverage, d64),
    }

    # threshold sensitivity
    taus = [0.02, 0.05, 0.1, 0.2, 0.5]
    labels = {}
    for name in ckpts:
        p0 = pre["checkpoints"][name]["pass"]["64"]
        c = pre["checkpoints"][name]["probes"]["coverage"]
        grid = {}
        for tp in taus:
            for tc in taus:
                grid[f"{tp}/{tc}"] = classify(p0, c, tp, tc)
        vals = list(grid.values())
        modal = max(set(vals), key=vals.count)
        labels[name] = {
            "grid": grid,
            "modal": modal,
            "modal_frac": vals.count(modal) / len(vals),
        }
    e3["threshold_sensitivity"] = labels

    result = {"E2_contextual_probes": e2, "E3_prediction": e3}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("E3 spearman support->dpass1:", e3["spearman_support_vs_delta_pass1"],
          "coverage->dpass1:", e3["spearman_coverage_vs_delta_pass1"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--pre", default="results/pre_rl.json")
    ap.add_argument("--grpo-summary", default="results/grpo_summary.json")
    ap.add_argument("--out", default="results/extensions.json")
    args = ap.parse_args()
    main(args.ckpt_dir, args.pre, args.grpo_summary, args.out)
