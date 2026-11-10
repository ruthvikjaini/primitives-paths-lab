# Research log — Day 17: primitives-paths-lab (2026-11-10)

## Trending-first check (2026-11-10, ~00:05 ET)

GitHub daily/weekly trending (gitdiscover, whatstrending.ai, upstar.dev, hotrepo.ai)
is again dominated by skill/prompt packs and agent frameworks/platforms, as on
every recent night:

- Skill packs: `janestreet/agent_skills` (+424/day), `cloudflare/security-audit-skill`
  (66K stars, was 1,848/day Nov 4–9), `mattpocock/skills` (318K), `anthropics/skills`.
- Agent frameworks/platforms: `morluto/rea` (446,888 stars, +3,218/day — reverse
  engineering with agents), `stablyai/orca` (fleet of coding agents),
  `vercel-labs/agent-desktop`, `meta/astryx` (design system), `milind-soni/OpenMausBot`.
- Code repos: `google-deepmind/mujoco` (16.9K, +574/day — physics engine, no
  ML-training mechanism to optimise in one CPU night), `ZhuLinsen/daily_stock_analysis`
  (68.9K — LLM stock analysis app), `synchopate/llama-modchip` (780 stars — runs a
  284B DeepSeek on a 64 GB CPU; needs the checkpoint), `FareedKhan-dev/kimi-k3-in-c`
  (11.5K — needs the 1.56 TB Kimi K3 checkpoint), `snap-stanford/memorilla` (759
  stars, +723 — latent semantic memory for LLMs; needs a real LLM to benchmark).
- New model releases (arsentev.ai): Reflection "Beam", Moonshot Kimi K3.1 Pro,
  Ricursive Ripple-65B, Lightricks LTX-2.5 — all need real checkpoints/GPU.

No trending repo has a genuine, measurable, one-night CPU weakness. Paper
fallback per the standing rule.

## Papers considered (arXiv, 5–9 Nov 2026)

- **Qu, "Primitives and Paths: Predicting What Post-Training Can Do from What
  the Base Model Already Has" (arXiv:2611.11260, submitted 8 Nov 2026, Oxford,
  workshop "Transitioning from Pre-Training to Post-Training") — CHOSEN.**
  Two-dimensional state space for post-trainability: (1) effective full-solution
  support p̂₀(x) = pass@K at the largest feasible K (Chen et al. unbiased
  estimator); (2) atomic primitive coverage ĉ(x) = min_j q_j^exec, the isolated
  execution probability of each primitive in the reference decomposition.
  Three regimes: amplification (both high), composition-limited (coverage high,
  support low), acquisition-limited (both low); high support + low coverage is a
  diagnostic-mismatch region, not a regime. Measurement protocol: reference
  decompositions, pass@k, isolated probes, handoff curve H_x(t) (supply first t
  steps of a verified reference trajectory, measure completion), composition gap
  G_comp(x;t) = H_x(t) − H_x(0). Experiments: synthetic composition on 5-digit
  strings, F = rotate left by 1, G = increment first digit by 3 mod 10;
  problems are 2–4-op sequences containing both F and G; programmatic verifier
  checks every intermediate state and the final answer. Three small LMs
  (Qwen3-1.7B, Gemma-3-1B, Qwen2.5-1.5B-Instruct); checkpoints M1 (F only),
  M1+C^(100/300), M2 (F+G, never mixed), M2+C^(100/300); 150 held-out problems,
  64 samples/problem; GRPO 150 updates, 3 seeds, reward requires a fully
  correct trajectory. Headline: primitive mastery alone does NOT guarantee RL
  can learn a composition — Qwen3 M2 has F/G = 1.00/1.00, pre-RL pass@64 = 0,
  post-RL pass@1/pass@64 = 0.000/0.000; Gemma M2 is the exception
  (0.476/0.500) because 6/2,000 training prompts were solved pre-RL despite
  zero held-out pass@64. Modest composition supervision that establishes
  nonzero support enables large gains: M1+C^(300) Qwen3 0.003/0.033 →
  0.172/0.376; M2+C^(100) Qwen3 0.064/0.180 → 0.452/0.482; M2+C^(300) Qwen3
  0.751/0.893 → 0.988/1.000 (amplification signature: Δpass@1 +0.237 >
  Δpass@64 +0.107). M1+C^(300) retains near-zero isolated-G accuracy
  (0.00–0.01) while using G inside complete trajectories — isolated probes can
  miss in-context skills (paper Sec. 6 limitation 1). Verified on arXiv
  abstract + full HTML 2026-11-10. No paper code read or used; independent
  implementation from the paper's equations and protocol.
- Zhai et al., "Does RL Expand the Capability Boundary of LLM Agents? A
  PASS@(k,T) Analysis" (arXiv:2604.14877) — resurfaced in Nov 9 news (AI
  Weekly alert), but the paper itself is v1 from 16 Apr 2026, not a new
  submission; held as context/citation, not the build. Its Table 1 (Cat C
  bridge: base 77, SFT 73, RL 81 of 100; RL\SFT 9 vs SFT\RL 1) is the
  counterpoint the Qu paper reconciles.
- Lu et al., "When Credit Turns Against Tools: Transition-Aware Credit
  Assignment in Agentic Reinforcement Learning" (arXiv:2611.11330, 8 Nov 2026)
  — TACA: tool-token imbalance → negative tool credit; Advantage Projection,
  Decision Contrast, Supervised Replay; +4.61pp over MT-GRPO on Qwen2.5-1.5B.
  Strong mechanism paper, held as a backup: same agentic-RL cluster as Days
  1/6/13/15, and the Qu framework question is fresher for this series.
- Hwang et al., "The Race Is Decided Early! A Kernel Law Predicts Lock-In in
  Majority-Vote Test-Time RL" (arXiv:2611.11165, 8 Nov 2026) — theory-heavy
  (kernel probe + closed form); a faithful CPU test would be thin. Rejected.
- Yu et al., "How Much Training-Inference Mismatch Can On-Policy Distillation
  Tolerate?" (arXiv:2611.11450, 8 Nov 2026) — too close to Day 15 (opd-lab).
  Rejected.
- Mitra, "Agent Self-Improvement with Frozen Weights: The Added Value and
  Limits of Failure Traces" (arXiv:2611.11342, 8 Nov 2026) — single-author,
  largely null results, needs a 9B model + teacher. Rejected.
- Held from earlier nights (unchanged): Liu "No Free Lunch in RLVR"
  (arXiv:2610.39295); Reward Budgeting (arXiv:2611.05942); KAPPO
  (arXiv:2611.03146). The Qu paper is the constructive answer to the
  No-Free-Lunch debate: whether RL teaches anything new depends on the
  base model's (support, coverage) state.

## Build plan

Reproduce Qu's experiment at tiny-transformer scale (1-layer, d=64, 4 heads,
NumPy, manual backprop, CPU-only), following the paper's checkpoint ladder
exactly: M1, M1+C100, M1+C300, M2, M2+C100, M2+C300, with SFT histories
controlling primitive exposure. Pre-RL protocol per Sec. 4: pass@1/pass@64
(Chen unbiased estimator, n=64, temp 0.7, 150 held-out problems), isolated F/G
probes (coverage = min), handoff curves, training-prompt support (the Gemma
M2 anomaly check). GRPO: binary reward on fully-correct trajectories, G=8,
clip 0.2, 150 updates, 3 seeds where time allows. Extensions (paper Sec. 6
limitations as prompts): (E1) handoff curves + composition gaps per
checkpoint — the instrument is proposed in Sec. 4 but no handoff numbers are
reported; (E2) in-context primitive probe (G after a supplied F prefix) vs
the isolated probe, testing whether the M1+C300 "G ≈ 0 in isolation but used
in trajectories" finding is a probe artifact; (E3) prospective-prediction test
— rank correlation between pre-RL (support, coverage) and post-RL Δpass@1
across checkpoints, plus threshold-sensitivity of the regime classification.

## Calibration notes (tiny-transformer scale, 2026-11-10)

The paper's checkpoints are 1.5–1.7B pretrained LMs; the reproduction is a
1-layer d=64 transformer trained from scratch, so SFT histories need
recalibration. The F primitive (rotate left, a copy) is learned in 1 epoch
(2000 examples, F=1.00). The G primitive (first digit +3 mod 10) shows a
sharp, grokking-like transition and is the calibration bottleneck:

| training | loss | F | G |
|---|---|---|---|
| F-only 2000x1 | 0.124 | 1.00 | 0.00 |
| G-only 4000x1 | 0.012 | 0.00 | 1.00 |
| FG 4000x1 | 0.226 | 0.99 | 0.10 |
| FG 4000x3 | 0.206 | 1.00 | 0.18 |
| FG 4000x5 | 0.000 | 1.00 | 1.00 |
| FG 8000x3 | 0.000 | 1.00 | 1.00 |
| FG 2000x5 | 0.002 | 1.00 | 0.99 |
| FG 3000x5 | 0.216 | 1.00 | 0.14 |
| FG 4000x3 lr 6e-3 | 0.128 | 0.71 | 0.64 |
| FG 4000x3 lr 1e-2 | 0.185 | 0.83 | 0.76 |

G learning is non-monotonic in data seen (2000x5 clicks, 3000x5 does not,
4000x5 clicks) and higher LR destabilises F — consistent with a sharp
phase transition for the modular-addition map rather than gradual
learning. Stage-1 therefore uses a probe-verified recipe (train until
isolated F/G >= 0.99, retrying with a fresh seed if needed) rather than a
fixed epoch count; the ladder structure (which histories see which
primitives/compositions) is unchanged from the paper. A primitive
curriculum (F-only -> G-only -> FG mixed) was tested and REJECTED: the
G-only stage catastrophically forgets F (F=0.00) and fails to learn G
(G=0.00), and the FG consolidation recovers F but not G. Direct FG joint
training clicks for 2 of 3 model seeds at 4000x5 (seeds 0, 1 click;
seed 2 does not), so the retry loop is the reliable route.

Stage-2 deviation (replay): 1 epoch over 100-300 composition examples
alone, at lr 1e-3, damages the fragile G circuit — M2+C300 fell to
F/G = 0.94/0.42 (M2+C100 survived at 1.00/1.00; M1+C300 kept F=1.00).
Stage 2 therefore adds single-op replay (2000 F for the M1 branch, 2000 F
+ 2000 G for the M2 branch): with replay, M2+C300 stays at F/G = 1.00/1.00
(replay 1000 gives 0.98/0.99). Composition exposure is unchanged (n
examples); replay only maintains the primitives, which the paper's 1.5B
LMs maintain for free. M1 replays F only, so M1 still never sees G
outside compositions.

Stage-2 deviation (epochs): at tiny scale, 1 epoch over 100-300 comp
examples establishes ZERO support (pre-RL pass@64 = 0.000 for every
checkpoint on the full 150-problem held-out set) — the from-scratch model
cannot learn the multi-state compositional format from a handful of
examples, where the paper's pretrained LMs can (their M2+C100 already has
pass@64 0.180). Stage 2 therefore runs 10 epochs over the same comp+replay
mixture (example counts unchanged: still 100/300), which is enough for the
model to learn the compositional format (multi-state responses with '='
separators) while the replay keeps the primitives intact. The true pre-RL
pass rates for the 10-epoch ladder are in results/pre_rl.json (measured
with scripts/measure_all.py on the full 150-problem set).
Note (bug caught during calibration): an early dose-response probe
reported M2+C300 pass@32 = 0.933 at 10 epochs, but it drew its 60
"held-out" problems as `[make_problem(default_rng(123)) for _ in range(60)]`
— a fresh RNG per problem, so all 60 were the SAME problem. That number
was one problem's solve rate, not a pass rate, and is discarded. The
library scripts (measure_all.py, run_grpo_grid.py) always draw problems
from a single shared RNG stream; demo.py and the handoff test had the same
fresh-RNG bug and were fixed.

GRPO gradient bugs (caught by a pilot run before the grid):
1. KL-sign bug. The original surrogate gradient used coefficient (w + beta),
   but d logp / d logits = onehot - probs = -(probs - onehot), so the KL
   term contributes -beta, not +beta. With the wrong sign the update
   maximised the KL: on a zero-signal checkpoint the KL climbed 0 -> 2.03
   in 5 updates with no reward at all. Correct coefficient is (w - beta).
2. No-signal drift. With all rewards equal, every advantage is zero, so the
   KL term alone drove the policy (Adam takes a full-size step in the
   direction of the tiny KL gradient). grpo_update now returns early without
   stepping when signal_frac == 0, so zero-support checkpoints stay exactly
   at their pre-RL state — the correct "no support, no learning" result.
3. Step size. The sign is a first-order property: at lr 1e-5..1e-4 a GRPO
   update raises the rewarded trajectory's logprob and lowers the
   unrewarded one's (verified directly); at lr 3e-3 a single Adam step
   overshoots the linear regime and the rewarded logprob falls. The grid
   therefore uses a small RL lr (see scripts/run_grpo_grid.py). The test
   suite now contains test_grpo_raises_rewarded_logprob, which checks the
   sign directly — the earlier params-moved test could not see a sign error.

## Sources

- Qu (2026), arXiv:2611.11260 — https://arxiv.org/abs/2611.11260
  (abstract + full HTML read 2026-11-10)
- Zhai et al. (2026), arXiv:2604.14877 — https://arxiv.org/abs/2604.14877
  (context: PASS@(k,T) and the capability-boundary debate)
- Yue et al. (2025), arXiv:2504.13837 — pass@k convergence on static reasoning
- Yuan et al. (2025), arXiv:2509.25123 — RL learns withheld compositions f(g(x))
- Chen et al. (2021), arXiv:2107.03374 — unbiased pass@k estimator
- GitHub trending: gitdiscover.org, whatstrending.ai, upstar.dev, hotrepo.ai
  (checked 2026-11-10); arXiv API cs.LG listing 7–10 Nov 2026.
