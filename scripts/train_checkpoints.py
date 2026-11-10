"""Build the SFT checkpoint ladder (paper Sec. 5).

M1:      F-only SFT.
M1+C100: M1 + continued SFT on 100 mixed compositions.
M1+C300: M1 + continued SFT on 300 mixed compositions.
M2:      F+G single-op SFT, never mixed.
M2+C100: M2 + continued SFT on 100 mixed compositions.
M2+C300: M2 + continued SFT on 300 mixed compositions.

Stage-1 sizes/epochs and stage-2 lr are set in CONFIG after pilot
calibration (see docs/RESEARCH.md). Checkpoints are saved as .npz under
--out (default checkpoints/); a manifest records the SFT config and the
isolated-probe accuracy of every checkpoint.
"""

import argparse
import json
import os
import time

import numpy as np

from primitives_paths.measure import probe_primitives
from primitives_paths.model import Transformer
from primitives_paths.task import VOCAB_SIZE
from primitives_paths.train import sft_examples, sft_train

# Calibrated so isolated F/G accuracy is ~1.00 for M1/M2 (see RESEARCH.md).
# The G primitive has a sharp, seed-dependent learning transition at this
# scale: joint FG training clicks (G=1.00) for some model seeds and not
# others, so M2 is trained with a probe-verified retry loop (below).
CONFIG = {
    "n_single": 4000,
    "epochs_m1": 1,
    "epochs_m2": 5,
    "lr_stage1": 3e-3,
    "epochs_stage2": 10,
    "lr_stage2": 1e-3,
    "n_replay": 2000,
    "batch": 64,
    "model_seed": 0,
    "data_seed": 1234,
    "probe_threshold": 0.99,
    "max_attempts": 5,
}

CHECKPOINTS = ["M1", "M1+C100", "M1+C300", "M2", "M2+C100", "M2+C300"]


def _train_stage1(kind, n_single, epochs, lr, batch, model_seed, data_seed,
                  threshold, max_attempts):
    """Train a stage-1 checkpoint, retrying with fresh model seeds until
    the isolated probes pass (the G transition is seed-dependent)."""
    for attempt in range(max_attempts):
        rng = np.random.default_rng(data_seed)
        model = Transformer(VOCAB_SIZE, seed=model_seed + attempt)
        ex = sft_examples(rng, kind, n_single=n_single)
        losses = sft_train(model, ex, epochs=epochs, batch=batch, lr=lr,
                           rng=rng)
        probes = probe_primitives(model, rng, n=200)
        ok = min(probes["F"], probes["G"]) >= threshold if kind == "FG" \
            else probes["F"] >= threshold
        print(f"  stage1 {kind} attempt {attempt} (seed {model_seed + attempt}): "
              f"loss {losses[-1]:.4f} probes F={probes['F']:.2f} "
              f"G={probes['G']:.2f} {'OK' if ok else 'retry'}", flush=True)
        if ok:
            return model, losses, probes, attempt
    raise RuntimeError(f"stage-1 {kind} never reached probe threshold "
                       f"{threshold} in {max_attempts} attempts")


def build(out_dir, config):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(config["data_seed"])
    manifest = {"config": config, "checkpoints": {}}

    # ---- stage 1: M1 (F only) and M2 (F+G, never mixed)
    stage1 = {}
    for name, kind, epochs in (("M1", "F", config["epochs_m1"]),
                               ("M2", "FG", config["epochs_m2"])):
        t0 = time.time()
        model, losses, probes, attempt = _train_stage1(
            kind, config["n_single"], epochs, config["lr_stage1"],
            config["batch"], config["model_seed"], config["data_seed"],
            config["probe_threshold"], config["max_attempts"])
        path = os.path.join(out_dir, f"{name}.npz")
        model.save(path)
        stage1[name] = model
        manifest["checkpoints"][name] = {
            "path": path, "stage": 1, "final_loss": losses[-1],
            "probes": probes, "seconds": round(time.time() - t0, 1),
            "attempts": attempt + 1,
        }
        print(f"{name}: loss {losses[0]:.3f}->{losses[-1]:.4f} "
              f"probes F={probes['F']:.2f} G={probes['G']:.2f} "
              f"({time.time()-t0:.0f}s, {attempt + 1} attempt(s))", flush=True)

    # ---- stage 2: continued SFT on n mixed compositions + single-op replay.
    # Replay is a scale deviation from the paper (documented in RESEARCH.md):
    # at tiny-transformer scale, 1 epoch over 100-300 composition examples
    # alone catastrophically damages the fragile G circuit (M2+C300 fell to
    # F/G = 0.94/0.42 without replay); the paper's 1.5B LMs do not forget in
    # 1 epoch. Replay maintains the primitives while the comp examples
    # establish support; composition exposure is unchanged (n examples).
    # M1 replays F only -- M1 must never see G outside compositions.
    for base, n in (("M1", 100), ("M1", 300), ("M2", 100), ("M2", 300)):
        name = f"{base}+C{n}"
        t0 = time.time()
        model = stage1[base].copy()
        replay_kind = "F" if base == "M1" else "FG"
        ex = (sft_examples(rng, "comp", n_comp=n)
              + sft_examples(rng, replay_kind,
                             n_single=config["n_replay"]))
        losses = sft_train(model, ex, epochs=config["epochs_stage2"],
                           batch=config["batch"], lr=config["lr_stage2"],
                           rng=rng)
        probes = probe_primitives(model, rng, n=200)
        path = os.path.join(out_dir, f"{name}.npz")
        model.save(path)
        manifest["checkpoints"][name] = {
            "path": path, "stage": 2, "base": base, "n_comp": n,
            "n_replay": config["n_replay"], "replay_kind": replay_kind,
            "final_loss": losses[-1], "probes": probes,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name}: loss {losses[0]:.3f}->{losses[-1]:.4f} "
              f"probes F={probes['F']:.2f} G={probes['G']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--n-single", type=int, default=CONFIG["n_single"])
    ap.add_argument("--epochs-m2", type=int, default=CONFIG["epochs_m2"])
    ap.add_argument("--lr2", type=float, default=CONFIG["lr_stage2"])
    args = ap.parse_args()
    cfg = dict(CONFIG)
    cfg["n_single"] = args.n_single
    cfg["epochs_m2"] = args.epochs_m2
    cfg["lr_stage2"] = args.lr2
    build(args.out, cfg)
