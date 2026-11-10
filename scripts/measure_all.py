"""Pre-RL measurement protocol for every checkpoint (paper Sec. 4).

For each checkpoint in checkpoints/manifest.json:
  - pass@1 / pass@64 (Chen unbiased estimator, n=64, temp 0.7) on 150
    held-out problems;
  - isolated F/G probes (greedy) and coverage = min(F, G);
  - handoff curve H(0..2) on held-out 3-op problems (n=16, temp 0.7);
  - training-prompt support: fraction of 256 RL training prompts solved
    at least once in 64 samples (the Gemma-M2 anomaly check).

Writes results/pre_rl.json.
"""

import argparse
import json
import os
import time

import numpy as np

from primitives_paths.measure import (
    evaluate_pass, handoff_curve, probe_primitives, training_support,
)
from primitives_paths.model import Transformer
from primitives_paths.task import VOCAB_SIZE, make_problem

N_HELDOUT = 150
N_TRAIN = 1000
N_TRAIN_SUPPORT = 256


def heldout_problems(seed=123):
    rng = np.random.default_rng(seed)
    return [make_problem(rng) for _ in range(N_HELDOUT)]


def train_problems(seed=456):
    rng = np.random.default_rng(seed)
    return [make_problem(rng) for _ in range(N_TRAIN)]


def main(ckpt_dir, out_path):
    with open(os.path.join(ckpt_dir, "manifest.json")) as f:
        manifest = json.load(f)
    heldout = heldout_problems()
    m3 = [p for p in heldout if len(p[1]) == 3]
    train = train_problems()
    train_sub = train[:N_TRAIN_SUPPORT]

    results = {"heldout_n": N_HELDOUT, "train_n": N_TRAIN, "checkpoints": {}}
    for name in ["M1", "M1+C100", "M1+C300", "M2", "M2+C100", "M2+C300"]:
        t0 = time.time()
        model = Transformer.load(os.path.join(ckpt_dir, f"{name}.npz"),
                                 VOCAB_SIZE)
        rng = np.random.default_rng(20240)
        ev = evaluate_pass(model, heldout, n_samples=64, temperature=0.7,
                           rng=rng)
        probes = probe_primitives(model, rng, n=200)
        hc = handoff_curve(model, m3, n_samples=16, temperature=0.7, rng=rng)
        ts = training_support(model, train_sub, n_samples=64,
                              temperature=0.7, rng=rng)
        results["checkpoints"][name] = {
            "pass": {str(k): v for k, v in ev["pass"].items()},
            "mean_success_rate": ev["mean_success_rate"],
            "probes": probes,
            "handoff": hc,
            "train_support": ts,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name}: pass@1 {ev['pass'][1]:.3f} pass@64 {ev['pass'][64]:.3f} "
              f"F/G {probes['F']:.2f}/{probes['G']:.2f} "
              f"H {hc[0]:.2f}/{hc[1]:.2f}/{hc[2]:.2f} "
              f"train-solved {ts['frac_solved']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="results/pre_rl.json")
    args = ap.parse_args()
    main(args.ckpt_dir, args.out)
