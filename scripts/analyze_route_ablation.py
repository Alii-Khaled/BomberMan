#!/usr/bin/env python3
"""Pool matched-seed E135 ablations and apply the fixed promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_arm(prefix: str, arm: str, seeds: list[int]):
    docs = []
    for seed in seeds:
        path = REPO / "results" / f"{prefix}_{arm}_s{seed}.json"
        with path.open() as fh:
            docs.append(json.load(fh))
    return docs


def pooled(docs):
    n = sum(int(d["n_rounds"]) for d in docs)
    scores = np.asarray([
        rd["Harvy"] for doc in docs for rd in doc["per_round"]
    ], dtype=float)
    sums = {}
    for key in ("round_wins", "kills_total", "suicides_total",
                "coins_total", "bombs_total", "crates_total"):
        sums[key] = sum(float(doc["summary"]["Harvy"][key]) for doc in docs)
    rank = sum(float(doc["summary"]["Harvy"]["mean_rank"])
               * int(doc["n_rounds"]) for doc in docs) / n
    return {
        "n": n, "scores": scores, "score_mean": float(scores.mean()),
        "win_rate": sums["round_wins"] / n, "mean_rank": rank,
        "kills": int(sums["kills_total"]),
        "suicides": int(sums["suicides_total"]),
        "coins": int(sums["coins_total"]),
        "bombs": int(sums["bombs_total"]),
        "crates": int(sums["crates_total"]),
    }


def comparison(candidate, control, seeds, cand_docs, ctl_docs, rng):
    diff = candidate["scores"] - control["scores"]
    idx = rng.integers(0, len(diff), size=(50000, len(diff)))
    boot = diff[idx].mean(axis=1)
    flips = rng.choice(np.asarray([-1.0, 1.0]), size=(50000, len(diff)))
    null = (flips * diff).mean(axis=1)
    delta = float(diff.mean())
    p = float((np.count_nonzero(np.abs(null) >= abs(delta)) + 1)
              / (len(null) + 1))
    seed_deltas = []
    for seed, cand_doc, ctl_doc in zip(seeds, cand_docs, ctl_docs):
        cs = float(cand_doc["summary"]["Harvy"]["score_mean"])
        bs = float(ctl_doc["summary"]["Harvy"]["score_mean"])
        seed_deltas.append({"seed": seed, "delta": cs - bs})
    ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
    gates = {
        "delta_at_least_0.25": delta >= 0.25,
        "ci_lower_above_zero": ci[0] > 0.0,
        "win_rate_not_lower": candidate["win_rate"] >= control["win_rate"],
        "suicides_within_5pct": candidate["suicides"] <= control["suicides"] * 1.05,
        "kills_not_lower": candidate["kills"] >= control["kills"],
        "at_least_3_of_4_seeds": sum(x["delta"] > 0 for x in seed_deltas) >= 3,
    }
    return {
        "score_delta": delta, "ci95": ci, "randomization_p": p,
        "outscore_tie_loss": [int(np.sum(diff > 0)), int(np.sum(diff == 0)),
                               int(np.sum(diff < 0))],
        "seed_deltas": seed_deltas, "gates": gates,
        "passes_all": bool(all(gates.values())),
    }


def public_summary(p):
    return {k: v for k, v in p.items() if k != "scores"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e135_qual")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--arms", nargs="+",
                    default=["control", "dynamic", "bodyblock", "combined"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(135)
    docs = {arm: load_arm(args.prefix, arm, args.seeds) for arm in args.arms}
    stats = {arm: pooled(docs[arm]) for arm in args.arms}
    ctl = stats["control"]
    result = {
        "prefix": args.prefix, "seeds": args.seeds,
        "arms": {arm: public_summary(stats[arm]) for arm in args.arms},
        "comparisons": {},
    }
    for arm in args.arms:
        if arm == "control":
            continue
        result["comparisons"][arm] = comparison(
            stats[arm], ctl, args.seeds, docs[arm], docs["control"], rng)

    print("arm          score    win   rank kills suic bombs  delta       CI95   pass")
    print("-" * 84)
    for arm in args.arms:
        s = stats[arm]
        if arm == "control":
            delta, ci, passed = 0.0, [0.0, 0.0], "-"
        else:
            c = result["comparisons"][arm]
            delta, ci, passed = c["score_delta"], c["ci95"], str(c["passes_all"])
        print(f"{arm:12s} {s['score_mean']:6.3f} {s['win_rate']:6.3f} "
              f"{s['mean_rank']:6.2f} {s['kills']:5d} {s['suicides']:4d} "
              f"{s['bombs']:5d} {delta:+6.3f} [{ci[0]:+.2f},{ci[1]:+.2f}] {passed}")
    if args.out:
        path = REPO / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(result, fh, indent=2)
        print("wrote", path.relative_to(REPO))


if __name__ == "__main__":
    main()
