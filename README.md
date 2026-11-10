# primitives-paths-lab

Can you predict what post-training will do from what the base model already
has? This repo reproduces and extends Qu (2026), *"Primitives and Paths:
Predicting What Post-Training Can Do from What the Base Model Already Has"*
(arXiv:2611.11260), at tiny-transformer scale: a 1-layer transformer trained
from scratch (NumPy, manual backprop, CPU-only) on a synthetic composition
task, with the paper's full checkpoint ladder and measurement protocol.

## The question

"Does RL teach language models new capabilities, or merely amplify ones
acquired during pre-training?" The paper argues the question has no answer
independent of the base model's *capability state*, and factors that state
into two measurable axes:

- **Effective full-solution support** — whether complete successful
  trajectories are accessible under the base policy (measured by pass@K at
  the largest feasible sampling budget K).
- **Atomic primitive coverage** — whether the constituent skills of a
  solution are individually executable when the burden of composing them is
  removed (measured by isolated single-operation probes; coverage is the
  minimum over primitives).

The two axes define three regimes: **amplification** (both high),
**composition-limited** (coverage high, support low), and
**acquisition-limited** (both low). High support with low coverage is a
diagnostic-mismatch region, not a regime — it means the decomposition or
the probes are wrong.

## The task (paper Sec. 5)

Five-digit strings; two primitives: `F` rotates the string left by one
position, `G` increments the first digit by 3 mod 10. A problem is a
sequence of 2–4 operations containing both `F` and `G`; a programmatic
verifier checks every intermediate state and the final answer. Example:

```
12345 F G = 23451 = 53451
```

## Checkpoints (paper Sec. 5)

| Checkpoint | SFT history |
|---|---|
| M1 | `F` only |
| M1+C100 | M1 + 100 mixed-composition examples |
| M1+C300 | M1 + 300 mixed-composition examples |
| M2 | `F` + `G` single-op, never mixed |
| M2+C100 | M2 + 100 mixed-composition examples |
| M2+C300 | M2 + 300 mixed-composition examples |

Each checkpoint is measured pre-RL (pass@1/pass@64 with n=64 samples at
temperature 0.7 on 150 held-out problems, isolated F/G probes, handoff
curve, training-prompt support) and then trained with outcome-based GRPO
(binary reward on fully-correct trajectories, group 8, clip 0.2, 150
updates, 3 seeds) and re-measured.

## Results

### Master table (150 held-out problems; post-RL is the mean of 3 seeds)

| Checkpoint | pre pass@1 | pre pass@64 | pre F / G | post pass@1 | post pass@64 | Δpass@1 | Δpass@64 |
|---|---|---|---|---|---|---|---|
| M1 | 0.000 | 0.000 | 1.00/0.00 | 0.000 | 0.000 | +0.000 | +0.000 |
| M1+C100 | 0.000 | 0.000 | 1.00/0.00 | 0.000 | 0.000 | +0.000 | +0.000 |
| M1+C300 | 0.000 | 0.000 | 1.00/0.00 | 0.000 | 0.004 | +0.000 | +0.004 |
| M2 | 0.000 | 0.000 | 1.00/1.00 | 0.000 | 0.000 | +0.000 | +0.000 |
| M2+C100 | 0.000 | 0.007 | 0.99/1.00 | 0.000 | 0.002 | -0.000 | -0.004 |
| M2+C300 | 0.037 | 0.233 | 1.00/1.00 | 0.208 | 0.304 | +0.170 | +0.071 |

### What the table says

- **The paper's headline reproduces.** M2 has both primitives mastered
  (F/G = 1.00/1.00) and zero full-solution support (pre pass@64 = 0.000),
  and GRPO gains nothing from it (0.000 → 0.000 over 150 updates, 3 seeds;
  every update saw zero reward signal and was skipped). Primitive mastery
  alone does not guarantee RL can learn a composition.
- **Only the checkpoint with real support gains.** M2+C300 is the only
  checkpoint with nonzero support (pre pass@1 0.037, pass@64 0.233; 30.9%
  of training prompts solved pre-RL), and it is the only one that improves
  substantially: Δpass@1 +0.170, Δpass@64 +0.071, with training reward
  0.066 → 0.224. The gain is an **amplification signature** —
  Δpass@1 is 2.4× Δpass@64, exactly the pattern the paper predicts for a
  high-support checkpoint (its M2+C300: Δpass@1 +0.237 > Δpass@64 +0.107).
- **Support is a threshold, not a slope.** M2+C100 has pass@64 0.007 —
  one problem in 150 solved once in 64 samples — and gains nothing
  (Δ −0.000/−0.004). At this scale 100 composition examples are below the
  threshold for establishing usable support, where the paper's pretrained
  LMs already gain from 100 (their M2+C100: 0.064/0.180 → 0.452/0.482).
  M1+C300 gains only a whisper (Δpass@64 +0.004), consistent with its
  1.2% training-prompt support: M1 never learns G, so it has almost no
  compositional trajectories for RL to amplify.
- **RL reshapes the primitives it was supposed to only amplify (E4).**
  M2+C300's isolated-F accuracy falls to 0.23 after GRPO (mean of 3 seeds;
  0.20/0.14/0.34) while G stays 1.00 — yet the model applies F correctly
  inside complete trajectories (verified by greedy generations: the F step
  of `12345 F G` is exactly `23451`, wrap-around included). Teacher-forced
  probabilities show the damage is localised: the correct 3rd/4th digits
  of an isolated F response drop to 0.05/0.00 probability, while the 1st,
  2nd, 5th digits and EOS stay at ~1.00. The isolated probe stops
  measuring the skill after RL has moved the policy — the mirror image of
  the paper's M1+C300 observation (G ≈ 0 in isolation, used in
  trajectories), and a caveat on the paper's coordinates: coverage is not
  stable under training.

### Extensions (measured)

- **E1 — Handoff curves.** H(t) rises with composition supervision:
  H(2) is 0.00 (M1, M2), 0.06 (M1+C100), 0.16 (M1+C300), 0.01 (M2+C100),
  0.19 (M2+C300). The composition gap G_comp(2) = H(2) − H(0) is 0.184 for
  M2+C300 and ≤ 0.16 elsewhere. A model that cannot complete a composition
  from a supplied 2-step prefix cannot be RL-amplified into one.
- **E2 — In-context primitive probe.** Contextual accuracy (apply one
  primitive to a supplied state) is 0.00/0.00 for M2 despite isolated
  F/G = 1.00/1.00, and 0.72/0.50 for M2+C300. Applying a primitive to a
  supplied state is itself a compositional skill that must be learned,
  not a free consequence of isolated mastery — and it tracks support
  (pass@64), not coverage. For M1+C300 the contextual G probe is 0.08
  against isolated G 0.00: the paper's "G used in trajectories" pattern
  does not appear pre-RL at this scale; it appears post-RL, for F (see
  above).
- **E3 — Prospective prediction.** Spearman rank correlation between
  pre-RL support (pass@64) and post-RL Δpass@1 is 0.180, and between
  coverage and Δpass@1 is 0.131. These are weak — not because the
  relationship is absent, but because only one checkpoint gains and the
  other five are separated by estimator noise (±0.0001 in the unbiased
  pass@k values), which scrambles the ranks of the near-zero gains. The
  cleaner statement is binary and exact: support > 0 ⟺ gain > 0 across
  all six checkpoints. Regime classification is stable: over a 5×5 grid
  of (τ_p, τ_c) thresholds, M1/M1+C100/M1+C300 are acquisition-limited,
  M2/M2+C100 composition-limited, and M2+C300 amplification (modal
  fraction 1.00, 1.00, 1.00, 1.00, 1.00, 0.80).

### Paper comparison (Qu 2026, Table 1 — Qwen3-1.7B / Gemma-3-1B / Qwen2.5-1.5B)

| Checkpoint | pre pass@1 | pre pass@64 | post pass@1 | post pass@64 |
|---|---|---|---|---|
| Qwen3 M2 | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen3 M2+C100 | 0.064 | 0.180 | 0.452 | 0.482 |
| Qwen3 M2+C300 | 0.751 | 0.893 | 0.988 | 1.000 |
| Gemma M2 | 0.000 | 0.000 | 0.476 | 0.500 |
| Qwen2.5 M2+C300 | 0.192 | 0.533 | 0.868 | 0.971 |

The paper's headline: primitive mastery alone does not guarantee RL can
learn a composition (Qwen3 M2: F/G = 1.00/1.00, pre-RL pass@64 = 0,
post-RL = 0), while modest composition supervision that establishes nonzero
support enables large gains. Gemma M2 is the exception: 6 of 2,000 training
prompts were solved pre-RL despite zero held-out pass@64 — held-out support
underestimates training-time support.

## Extensions

- **E1 — Handoff curves.** The paper proposes the handoff curve H(t)
  (supply the first t steps of a verified reference trajectory, measure
  completion) and the composition gap G_comp(t) = H(t) − H(0) as a
  composition diagnostic, but reports no handoff numbers. We measure the
  curve for every checkpoint.
- **E2 — In-context primitive probe.** The paper's M1+C300 checkpoints
  retain near-zero isolated-G accuracy while using G inside complete
  trajectories (Sec. 6 limitation). We compare isolated G/F accuracy with
  contextual accuracy (handoff t=1 completion on 2-op problems, where the
  only remaining step is one primitive applied to a supplied state) to test
  whether the probe, not the skill, is missing.
- **E3 — Prospective prediction.** Spearman rank correlation between the
  pre-RL coordinates (support, coverage) and post-RL gain across the six
  checkpoints, plus regime-classification stability over a grid of
  (τ_p, τ_c) thresholds.
- **E4 — Post-RL primitive divergence (new).** After GRPO, M2+C300's
  isolated-F accuracy collapses (1.00 → 0.23) while its compositional
  success rises — the model applies F correctly inside trajectories but
  fails it in isolation. RL moves the policy off the probe's distribution,
  so post-RL coverage is not the same quantity as pre-RL coverage.

## How to run

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/train_checkpoints.py   # build the ladder
PYTHONPATH=src python scripts/measure_all.py        # pre-RL protocol
PYTHONPATH=src python scripts/run_grpo_grid.py      # GRPO, 150 updates x 3 seeds
PYTHONPATH=src python scripts/extensions.py         # E2 + E3
PYTHONPATH=src python scripts/evaluate_all.py       # -> results/measured.json
PYTHONPATH=src python scripts/crossfoot.py          # verify README figures
PYTHONPATH=src python scripts/demo.py               # quick end-to-end demo
PYTHONPATH=src pytest tests/ -q
```

Checkpoints (`.npz`) live in `checkpoints/` (gitignored); all measured
numbers live in `results/*.json`. The demo trains a small M2 and M2+C100
and runs a short GRPO pass in a couple of minutes.

## Sources

- Qu, A. (2026). *Primitives and Paths: Predicting What Post-Training Can
  Do from What the Base Model Already Has.* arXiv:2611.11260.
  https://arxiv.org/abs/2611.11260
- Yue et al. (2025). *Does Reinforcement Learning Really Incentivize
  Reasoning Capacity in LLMs Beyond the Base Model?* arXiv:2504.13837.
- Yuan et al. (2025). *From f(x) and g(x) to f(g(x)): LLMs Learn New
  Skills in RL by Composing Old Ones.* arXiv:2509.25123.
- Zhai et al. (2026). *Does RL Expand the Capability Boundary of LLM
  Agents? A PASS@(k,T) Analysis.* arXiv:2604.14877.
- Chen et al. (2021). *Evaluating Large Language Models Trained on Code*
  (unbiased pass@k estimator). arXiv:2107.03374.

## Limitations

- Tiny from-scratch transformer, not a pretrained 1.5B LM: SFT histories
  need more epochs than the paper's (see `docs/RESEARCH.md` for the
  calibration, including the sharp, seed-dependent transition in learning
  the G primitive), and absolute numbers are not comparable to the paper's
  — the comparison is about the *pattern* across checkpoints. Two
  paper results do not reproduce at this scale: the paper's M2+C100 gains
  a lot from 100 composition examples (0.064/0.180 → 0.452/0.482) while
  ours gains nothing (100 examples are below the tiny-scale support
  threshold), and the paper's M1+C300 gains (0.003/0.033 → 0.172/0.376)
  while ours gains almost nothing (+0.004 pass@64), because M1 never
  learns G and has ~1% training-prompt support.
- 150 held-out problems and 3 GRPO seeds: small-sample noise is real
  (pass@64 has a ~4-point standard error per checkpoint at these sizes).
  The unbiased pass@k estimator also leaves ±0.0001-level noise on
  near-zero gains, which scrambles rank correlations among the
  non-gaining checkpoints (see E3).
- The composition task has exactly two primitives and one valid
  decomposition per problem; natural reasoning tasks admit many.
- GRPO uses lr 1e-4, calibrated on this model: lr 1e-3 destroys the
  fragile F primitive within 30 updates (F 1.00 → 0.31), and the
  surrogate gradient's KL term must carry a minus sign (see
  `docs/RESEARCH.md` — the original code had the KL sign flipped and the
  policy drifted away from the reference with zero reward signal).
