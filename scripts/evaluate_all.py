"""Aggregate pre-RL, GRPO, and extension results into results/measured.json.

The master table mirrors the paper's Table 1 (pre/post pass@1, pass@64,
isolated F/G, Delta pass@1, Delta pass@64) plus the handoff curve, the
training-prompt support, and the extension results.
"""

import argparse
import json
import os

import numpy as np

CKPTS = ["M1", "M1+C100", "M1+C300", "M2", "M2+C100", "M2+C300"]
SEEDS = (0, 1, 2)


def main(results_dir):
    with open(os.path.join(results_dir, "pre_rl.json")) as f:
        pre = json.load(f)
    with open(os.path.join(results_dir, "grpo_summary.json")) as f:
        summ = json.load(f)
    ext = {}
    ext_path = os.path.join(results_dir, "extensions.json")
    if os.path.exists(ext_path):
        with open(ext_path) as f:
            ext = json.load(f)

    table = {}
    for name in CKPTS:
        p = pre["checkpoints"][name]
        posts1 = [summ[f"{name}_seed{s}"]["post_pass"]["1"] for s in SEEDS
                  if f"{name}_seed{s}" in summ]
        posts64 = [summ[f"{name}_seed{s}"]["post_pass"]["64"] for s in SEEDS
                   if f"{name}_seed{s}" in summ]
        post_probes = [summ[f"{name}_seed{s}"]["post_probes"] for s in SEEDS
                       if f"{name}_seed{s}" in summ]
        table[name] = {
            "pre_pass1": p["pass"]["1"],
            "pre_pass64": p["pass"]["64"],
            "pre_F": p["probes"]["F"],
            "pre_G": p["probes"]["G"],
            "pre_coverage": p["probes"]["coverage"],
            "pre_handoff": {k: v for k, v in p["handoff"].items()
                            if k.isdigit()},
            "pre_gap1": p["handoff"]["gap1"],
            "pre_gap2": p["handoff"]["gap2"],
            "pre_train_frac_solved": p["train_support"]["frac_solved"],
            "pre_train_n_solved": p["train_support"]["n_solved"],
            "post_pass1_mean": float(np.mean(posts1)),
            "post_pass1_seeds": posts1,
            "post_pass64_mean": float(np.mean(posts64)),
            "post_pass64_seeds": posts64,
            "post_F_mean": float(np.mean([q["F"] for q in post_probes])),
            "post_G_mean": float(np.mean([q["G"] for q in post_probes])),
            "delta_pass1": float(np.mean(posts1)) - p["pass"]["1"],
            "delta_pass64": float(np.mean(posts64)) - p["pass"]["64"],
            "reward_first25_mean": float(np.mean(
                [summ[f"{name}_seed{s}"]["reward_first25"] for s in SEEDS
                 if f"{name}_seed{s}" in summ])),
            "reward_last25_mean": float(np.mean(
                [summ[f"{name}_seed{s}"]["reward_last25"] for s in SEEDS
                 if f"{name}_seed{s}" in summ])),
        }
    measured = {
        "heldout_n": pre["heldout_n"],
        "train_n": pre["train_n"],
        "table": table,
        "extensions": ext,
    }
    with open(os.path.join(results_dir, "measured.json"), "w") as f:
        json.dump(measured, f, indent=2)
    for name in CKPTS:
        r = table[name]
        print(f"{name}: pre {r['pre_pass1']:.3f}/{r['pre_pass64']:.3f} "
              f"F/G {r['pre_F']:.2f}/{r['pre_G']:.2f} -> post "
              f"{r['post_pass1_mean']:.3f}/{r['post_pass64_mean']:.3f} "
              f"(d {r['delta_pass1']:+.3f}/{r['delta_pass64']:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    main(args.results_dir)
