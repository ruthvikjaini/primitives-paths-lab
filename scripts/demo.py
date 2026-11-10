"""Quick demo: build M2 once, then show the M2 and M2+C100 branches
(pre-RL coordinates and a short GRPO run each) in the (support, coverage)
state space.

Runs in a couple of minutes on CPU (stage-1 SFT dominates). For the full
grid see scripts/train_checkpoints.py, scripts/measure_all.py and
scripts/run_grpo_grid.py.
"""

import os
import sys

import numpy as np

# Allow `python scripts/demo.py` from the project root (see run_grpo_grid.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives_paths.measure import evaluate_pass, probe_primitives
from primitives_paths.model import Transformer
from primitives_paths.task import VOCAB_SIZE, make_problem
from primitives_paths.train import grpo_train, sft_examples, sft_train
from scripts.train_checkpoints import _train_stage1


def _show(model, heldout, train, name, rng, grpo_updates=20):
    pre = evaluate_pass(model, heldout, n_samples=16, temperature=0.7,
                        rng=rng)
    pr = probe_primitives(model, rng, n=100)
    print(f"{name}: pre pass@1 {pre['pass'][1]:.3f} "
          f"pass@16 {pre['pass'][16]:.3f} "
          f"F/G {pr['F']:.2f}/{pr['G']:.2f}", flush=True)
    ref = model.copy()
    grpo_train(model, ref, train, updates=grpo_updates, batch=8, group=8,
               lr=1e-3, beta=0.02, temperature=1.0, rng=rng, log_every=0)
    post = evaluate_pass(model, heldout, n_samples=16, temperature=0.7,
                         rng=rng)
    print(f"{name}: post pass@1 {post['pass'][1]:.3f} "
          f"pass@16 {post['pass'][16]:.3f} "
          f"(Delta pass@1 {post['pass'][1]-pre['pass'][1]:+.3f})",
          flush=True)


def main():
    rng = np.random.default_rng(0)
    hrng = np.random.default_rng(123)
    heldout = [make_problem(hrng) for _ in range(60)]
    trng = np.random.default_rng(456)
    train = [make_problem(trng) for _ in range(200)]

    # M2 via the probe-verified stage-1 recipe (G has a sharp transition).
    m2, _, probes, attempt = _train_stage1(
        "FG", 2000, 5, 3e-3, 64, 0, 1234, 0.99, 3)
    print(f"M2 built in {attempt + 1} attempt(s), "
          f"probes F={probes['F']:.2f} G={probes['G']:.2f}", flush=True)

    _show(m2.copy(), heldout, train, "M2", rng)

    # M2+C100: the same M2 plus 100 composition examples, 1 epoch.
    m2c = m2.copy()
    sft_train(m2c, sft_examples(rng, "comp", n_comp=100), epochs=1,
              batch=64, lr=1e-3, rng=rng)
    _show(m2c, heldout, train, "M2+C100", rng)
    print("demo OK")


if __name__ == "__main__":
    main()
