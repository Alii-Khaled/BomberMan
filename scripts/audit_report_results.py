#!/usr/bin/env python3
"""Read-only audit of the E130 and final-agent numbers quoted in the report.

Run from the repository root with ``python3 -B scripts/audit_report_results.py``.
The required results and D1 BC weights are tracked under ``results/``.
This audit reads stored records and does not play new matches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("final", "neutral", "search_off", "bc")
FIELDS = ("g1", "strong", "archetypes")
SEEDS = 10
ROUNDS = 20


def get_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_e130():
    """Check exact identity of non-runtime E130 records for five field legs."""
    matched = 0
    for field in ("umix", "ucow", "ubom", "urus", "urac"):
        for seed in (0, 1):
            prefix = ROOT / "results"
            arm = get_json(prefix / f"tourney_e130_arm_{field}_s{seed}.json")
            control = get_json(prefix / f"tourney_e130_ctl_{field}_s{seed}.json")
            assert set(arm) == set(control)
            assert [key for key in arm if arm[key] != control[key]] == ["runtime_s"], (field, seed)
            matched += len(arm["per_round"])
    assert matched == 520, matched
    print("E130: 520 evaluation-only rounds per arm; all recorded non-runtime fields match")


def verify_final():
    summary = get_json(ROOT / "results/report_ablation/summary_s10_n20.json")
    assert summary["seeds"] == SEEDS and summary["rounds_per_seed"] == ROUNDS
    final_hash = digest(ROOT / "agent_code/Harvey/my-saved-model.pt")
    bc_hash = digest(ROOT / "results/arbiter_ng_d1.pt")
    assert final_hash == digest(ROOT / "results/e108rl.pt.ep225")
    data = {}
    for arm in ARMS:
        expected_hash = bc_hash if arm == "bc" else final_hash
        for field in FIELDS:
            for seed in range(SEEDS):
                path = ROOT / "results/report_ablation" / f"{arm}_{field}_s{seed}_n20.json"
                source = get_json(path)
                assert (source["arm"], source["field"], source["seed"], source["n_rounds"]) == (
                    arm, field, seed, ROUNDS
                ), path
                assert source["weights_sha256"] == expected_hash, path
                records = source["records"]
                assert len(records) == ROUNDS, path
                for row in records:
                    assert row["scores"]["Harvey"] == row["coins"] + 5 * row["kills"], path
                data[arm, field, seed] = records

    rng = np.random.default_rng(92713)
    for field in FIELDS:
        for seed in range(SEEDS):
            boards = [r["board_sha256"] for r in data["final", field, seed]]
            for arm in ARMS[1:]:
                assert boards == [r["board_sha256"] for r in data[arm, field, seed]], (
                    arm, field, seed
                )
        for arm in ARMS:
            rows = [r for seed in range(SEEDS) for r in data[arm, field, seed]]
            mean = float(np.mean([r["scores"]["Harvey"] for r in rows]))
            reported = summary["cohorts"][f"{arm}/{field}"]
            assert np.isclose(mean, reported["score"]), (arm, field)
            if arm == "final":
                opponents = {name: float(np.mean([r["scores"][name] for r in rows]))
                             for name in rows[0]["scores"] if name != "Harvey"}
                invalid = sum(r["invalid"] for r in rows)
                calls = sum(len(r["times_s"]) for r in rows)
                assert (invalid, calls) == {
                    "g1": (763, 49563), "strong": (662, 45523),
                    "archetypes": (273, 72604),
                }[field]
                print(f"{field}: final score {mean:.3f}; opponents {opponents}; "
                      f"invalid {invalid}/{calls} ({invalid / calls:.2%})")
        for arm in ARMS[1:]:
            blocks = np.asarray([
                np.mean([a["scores"]["Harvey"] - b["scores"]["Harvey"]
                         for a, b in zip(data["final", field, seed],
                                         data[arm, field, seed])])
                for seed in range(SEEDS)
            ])
            draws = rng.integers(0, SEEDS, size=(10000, SEEDS))
            interval = np.percentile(blocks[draws].mean(axis=1), [2.5, 97.5])
            recorded = summary["contrasts"][f"final-minus-{arm}/{field}"]
            assert np.allclose(blocks, recorded["seed_block_deltas"])
            assert np.isclose(blocks.mean(), recorded["score_delta"])
            assert np.allclose(interval, recorded["bootstrap_95_seed_blocks"])
            print(f"  final minus {arm}: {blocks.mean():+.3f}, "
                  f"seed-block 95% [{interval[0]:+.3f}, {interval[1]:+.3f}]")
    print("PASS: 120 complete cohorts, 2,400 rounds, matched starting boards and nine intervals")


if __name__ == "__main__":
    verify_e130()
    verify_final()
