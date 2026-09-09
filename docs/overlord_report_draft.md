# Overlord report threads (DRAFT — refine in own style before submission)

Covers the Overlord CNN agent for final-report §4 (Methods), §5
(Training) and §6 (Experiments). Source ledger: `docs/experiments.md`
entries E19, E22–E32, E38–E45. Main author: Ali Mahbob (per §9 author
marks — confirm/adjust per team split). Sentinel (§4 first-model) and
reaper (third model) are covered elsewhere; cross-references marked.
 companion figures listed at the end.

---

## §4.2 Methods — Overlord: spatial CNN with exact safety mask

*Main author: Ali Mahbob.*

While sentinel compresses the board into a 46-dimensional handcrafted
feature vector, overlord learns spatial structure directly: a ResNet
encoder (stem-64→96 channels, two residual blocks, stride-2 downsampling
17×17 → 9×9) followed by global average *and* global max pooling
(GAP+GMP), concatenated with an 8-dimensional scalar head (step,
bombs-left, crates, coins, opponents, escape flag, crates-hit,
opps-hit), a 512-unit fully connected layer, and a dueling head (value +
advantage) plus an auxiliary danger-prediction head (mean danger at t+1,
weight 0.1 for dense gradient). Total ~2M parameters (vs sentinel's MLP
and reaper's 230K). Both dueling heads are zero-initialized, so Q starts
near zero and the heuristic prior drives early play while the learned Q
grows from scratch — the same cold-start discipline as sentinel.

Decision-making mirrors sentinel's two-layer structure deliberately
(shared `safety.py` core): an exact, learning-free safety mask plus a
heuristic prior, with the CNN contributing `Q_WEIGHT × Q` on top.
The safety solver replicates the engine exactly (wall-aware blast,
timer-based danger map over an 8-step horizon, time-expanded BFS
escape search, hypothetical-bomb escape test `dist_hyp`, corridor
discipline, and the margin rule forbidding bombs with `dist_hyp > 3`).
Three facts established later (§6: E27, E42) justify this architecture:
the transplanted margin gate holds with zero ranking failures on CNN
dynamics, per-plant lethality sits at safe levels (1.4–2.7%), and an
intent audit proved the mask vetoes *zero* productive bombs — the
bottleneck is intent generation, never freedom. The shipped weight is
Q_WEIGHT = 1.0 (E30 sweep: 0.2 → 3.86, 0.5 → 3.25, 1.0 → 3.99 screen,
pooled 3.79 — calibrated small-Q wants authority, the opposite of the
diverged-Q finding on sentinel).

Training is N-step (N=5, covering the 4-step bomb timer inside the
return window — the sentinel N=3 scar) double-DQN with prioritized
replay, Huber loss (δ=1) on both heads after sentinel's MSE detonation
(Stage 4: loss median 480 → 74k), tuned AdamW, gradient clip 1.0, AMP
autocast on CUDA with channels-last tensors and cudnn autotuning
(~92% GPU duty on an L40S). Exploration is epsilon-greedy 1.0→0.05
with safety-masked sampling; a later fix (E31/C2) counts environment
steps instead of gradient steps after discovering the schedule was
silently coupled to the update config (~800 updates/round burned a
100k decay in ~125 rounds). Reward shaping mirrors sentinel's table
(coins, kills, crates, survival, movement pulls) with balanced ± pairs;
two shaping additions were tried and *rejected with evidence*
(crate-approach pull E28/E29, multi-crate bonus E43/E44 — §6).

## §5.2 Training — Overlord: validation curriculum and refocus

*Main author: Ali Mahbob.*

Overlord trained a short validation curriculum (O1s 200 coin-heaven →
O2s 300 solo classic → O3s 300 hunt vs peaceful+collector → O4s 500 vs
3× rule_based + frozen eval ≈ 1300 rounds), reusing sentinel's
stage-reset convention (archive best, clear cross-regime EMA, epsilon
re-warm) and per-round full-payload checkpoints (loss ≤5 rounds on the
two session kills that hit the program). O1s reached 44.5 coins/round
(CNN-from-pixels matching the MLP's 48.6 feature-engineered baseline).
O2s showed the classic regime shock (44.5 → 1.3 coins) with healthy
optimization (loss 0.0016 → 0.0004, `|w|` 40→42, zero suicides in 300
rounds). O3s produced the best hunting either agent had shown
(0.71 kills/round at 0.17 suicide vs a bombing opponent). O4s + frozen
eval passed the Task-4 gate: **3.70 vs best rule_based 3.64**
(0.47 kills, 0.33 suicide) — and a snapshot bake-off later found O3s
weights even stronger (**3.86**, shipped with Q1.0 at **3.79 pooled**).

The watcher-gated full 5250-round curriculum auto-launched on the
passing gate but was stopped after ~110 coin-heaven rounds: navigation
was already ceiling, and the program refocused on weakpoints (economy
1.35 vs 2.4 coins at equal bomb volume — the E15 placement disease,
confirmed on CNN by forensics). What followed is documented as a
sequence of falsified hypotheses rather than a march: crate-approach
pull (W1 coins 1.09, reverted), 900-round focused program (frozen
−0.52, drift), CHAMP single-change retune (−0.50), crate-light regime
(+200% in-regime, zero classic transfer, hunt dulled). Each cost little
because per-round checkpoints, kill-switch guards, and pre-registered
rollback triggers bounded every bet. The standing policy has been
frozen since the bake-off; all later work is frozen-only screens and
audits.

## §6.3 Experiments — Overlord results and negative results

*Main author: Ali Mahbob.*

**Gate tables.** Frozen 100-round CPU vs 3× rule_based: validation 3.70
(0.47 kills, 0.33 suicide) beats best-rb 3.64; bake-off winner o3sbest
3.86 screen / 3.59 pooled; shipped o3sbest×Q1.0 **3.79 pooled**
(0.48 kills, 0.36 suicide). Warden-mix 60rd: 3.17 vs warden 4.13
(gap halved from 2.40); sentinel-mix: 3.12 vs sentinel 3.05
(head-to-head win over our own champion).

**Audit tables (methods core).** Death attribution on the frozen policy
(E26, E14a protocol, 60 rounds, hooks reverted): 69% own-bomb vs 31%
enemy — own-bomb dominated like sentinel's 78%. Escape-solver audit
(E27, 1099 plants): zero `can_escape=False` plants, zero dist>3
plants, lethality 1.4% (dist-2) / 2.7% (dist-3) — transplant validated,
no change (tightening to ≥3 would gut 62% of bombing for 0.30
deaths/round). Intent audit (E42, 915 intents): **zero vetoed** —
the mask is exonerated; 68% of intents hit zero crates, so the
bottleneck is intent generation.

**Negative-results gallery (each with mechanism).** Heuristic grid
(E31: 11 variants, top-3 all falsified at 100×2 — including a same-seed
4.25→3.17 collapse that upgrades the E09 noise law: 40-round screens
are nearly worthless with unseeded agent RNG). TTA symmetry ensemble
(E38: 8.4 ms/step, but averaging dilutes orientation-specific features
the unequivocated CNN learned — train-time augmentation ≠ test-time
averaging). Late-hunt veto (E39: 2.95, −44% kills — exact replay of
sentinel's E08). Crate pull, multi-crate bonus, openness bonus (volume
without efficiency every time; openness held crates/bomb at exactly
0.68 while costing kills and safety). Same-regime volume (E29/CHAMP/W0:
−0.5 frozen thrice — argued as the fourth occurrence of the E12
pattern). The invariant across all six failures — frozen economy
±0.1 on crates — is itself reported as the finding that redirects
future work to demonstration (imitation) over scalars.

**Figure list.** (1) O1→O4 training curves (coins/kills/suicide/EMA +
loss/wnorm twin axes, with regime-shock annotation). (2) Frozen gate
bar chart (ship vs best-rb vs warden, both seeds). (3) P(death|plant)
by dist_hyp, sentinel-vs-overlord side-by-side. (4) Bake-off +
Q-sweep table. (5) Drift comparison (4 occurrences). (6) Intent-vs-plant
funnel (915/915/0).
