"""GRPO grid: every checkpoint x seed (paper Sec. 5: 150 updates, 3 seeds).

Binary reward on fully-correct trajectories, group G=8, clip 0.2,
k1 KL beta=0.02 against the checkpoint reference, batch 16 prompts/update,
temperature 1.0, lr 1e-4 (calibrated: lr 1e-3 destroys the fragile primitives
in 30 updates — F 1.00 -> 0.31 — while 1e-4 raises reward and pass@1 with
primitives intact; see docs/RESEARCH.md). After training, each run is
evaluated on the 150 held-out problems (pass@1/pass@64, n=64, temp 0.7)
and isolated probes.

Writes results/grpo_<ckpt>_seed<s>.json per run and
results/grpo_summary.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Allow `python scripts/run_grpo_grid.py` from the project root: the
# script's directory is on sys.path, not the project root, so make the
# root importable for the `scripts.measure_all` import below.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives_paths.measure import evaluate_pass, probe_primitives
from primitives_paths.model import Transformer
from primitives_paths.task import VOCAB_SIZE
from primitives_paths.train import grpo_train
from scripts.measure_all import heldout_problems, train_problems

CHECKPOINTS = ["M1", "M1+C100", "M1+C300", "M2", "M2+C100", "M2+C300"]


def run_one(ckpt_dir, name, seed, updates, batch, group, lr, beta,
            out_dir):
    t0 = time.time()
    model = Transformer.load(os.path.join(ckpt_dir, f"{name}.npz"), VOCAB_SIZE)
    ref = model.copy()
    rng = np.random.default_rng(1000 + seed)
    train = train_problems()
    stats = grpo_train(model, ref, train, updates=updates, batch=batch,
                       group=group, lr=lr, beta=beta, temperature=1.0,
                       rng=rng, log_every=50)
    heldout = heldout_problems()
    ev = evaluate_pass(model, heldout, n_samples=64, temperature=0.7, rng=rng)
    probes = probe_primitives(model, rng, n=200)
    model.save(os.path.join(out_dir, f"grpo_{name}_seed{seed}.npz"))
    result = {
        "checkpoint": name, "seed": seed, "updates": updates,
        "batch": batch, "group": group, "lr": lr, "beta": beta,
        "final_reward": stats[-1]["mean_reward"],
        "reward_first25": float(np.mean([s["mean_reward"] for s in stats[:25]])),
        "reward_last25": float(np.mean([s["mean_reward"] for s in stats[-25:]])),
        "mean_signal_frac": float(np.mean([s["signal_frac"] for s in stats])),
        "post_pass": {str(k): v for k, v in ev["pass"].items()},
        "post_probes": probes,
        "seconds": round(time.time() - t0, 1),
        "stats": stats,
    }
    with open(os.path.join(out_dir, f"grpo_{name}_seed{seed}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"{name} seed{seed}: reward {result['reward_first25']:.3f}->"
          f"{result['reward_last25']:.3f} pass@1 {ev['pass'][1]:.3f} "
          f"pass@64 {ev['pass'][64]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return result


def main(ckpt_dir, out_dir, updates, seeds, batch, group, lr, beta,
         only=None):
    os.makedirs(out_dir, exist_ok=True)
    names = only or CHECKPOINTS
    summary = {}
    for name in names:
        for seed in seeds:
            r = run_one(ckpt_dir, name, seed, updates, batch, group, lr,
                        beta, out_dir)
            summary[f"{name}_seed{seed}"] = {
                k: r[k] for k in ("checkpoint", "seed", "reward_first25",
                                  "reward_last25", "mean_signal_frac",
                                  "post_pass", "post_probes", "seconds")
            }
    with open(os.path.join(out_dir, "grpo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    main(args.ckpt_dir, args.out_dir, args.updates, args.seeds, args.batch,
         args.group, args.lr, args.beta, args.only)
