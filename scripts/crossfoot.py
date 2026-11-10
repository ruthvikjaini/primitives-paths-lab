"""Cross-foot the README against the measured results.

Recomputes the master table in results/measured.json from the raw
results/pre_rl.json and results/grpo_summary.json, checks it matches,
then checks that every master-table value (3-decimal formatting) appears
in README.md. Exits non-zero on any mismatch.
"""

import json
import math
import os
import sys

RESULTS = "results"
CKPTS = ["M1", "M1+C100", "M1+C300", "M2", "M2+C100", "M2+C300"]
SEEDS = (0, 1, 2)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def main():
    errors = []
    with open(os.path.join(RESULTS, "pre_rl.json")) as f:
        pre = json.load(f)
    with open(os.path.join(RESULTS, "grpo_summary.json")) as f:
        summ = json.load(f)
    with open(os.path.join(RESULTS, "measured.json")) as f:
        measured = json.load(f)

    for name in CKPTS:
        p = pre["checkpoints"][name]
        posts1 = [summ[f"{name}_seed{s}"]["post_pass"]["1"] for s in SEEDS]
        posts64 = [summ[f"{name}_seed{s}"]["post_pass"]["64"] for s in SEEDS]
        row = measured["table"][name]
        checks = {
            "pre_pass1": p["pass"]["1"],
            "pre_pass64": p["pass"]["64"],
            "pre_F": p["probes"]["F"],
            "pre_G": p["probes"]["G"],
            "pre_coverage": p["probes"]["coverage"],
            "post_pass1_mean": sum(posts1) / len(posts1),
            "post_pass64_mean": sum(posts64) / len(posts64),
            "delta_pass1": sum(posts1) / len(posts1) - p["pass"]["1"],
            "delta_pass64": sum(posts64) / len(posts64) - p["pass"]["64"],
            "pre_train_frac_solved": p["train_support"]["frac_solved"],
        }
        for k, v in checks.items():
            if not close(row[k], v):
                errors.append(f"{name}.{k}: measured {row[k]} != recomputed {v}")
        # handoff gaps recompute from the curve
        h = p["handoff"]
        if not close(row["pre_gap1"], h["1"] - h["0"]):
            errors.append(f"{name}.pre_gap1 mismatch")
        if not close(row["pre_gap2"], h["2"] - h["0"]):
            errors.append(f"{name}.pre_gap2 mismatch")

    # README must contain every master-table value at 3 decimals
    with open("README.md") as f:
        readme = f.read()
    for name in CKPTS:
        row = measured["table"][name]
        for k in ("pre_pass1", "pre_pass64", "post_pass1_mean",
                  "post_pass64_mean", "delta_pass1", "delta_pass64"):
            s = f"{row[k]:.3f}"
            if s not in readme:
                errors.append(f"README missing {name}.{k} = {s}")

    if errors:
        print("CROSSFOOT FAIL")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"CROSSFOOT PASS: {len(CKPTS)} rows recomputed from raw results; "
          "all master-table values present in README.md")


if __name__ == "__main__":
    main()
