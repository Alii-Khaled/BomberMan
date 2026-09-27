#!/usr/bin/env python3
"""Run the four-way route/body-block ablation in alternating order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
MODEL = REPO / "agent_code" / "Harvey" / "my-saved-model.pt"

ARMS = {
    "control": ("0", "0", "0"),
    "dynamic": ("0", "1", "0"),
    "bodyblock": ("0", "0", "1"),
    "combined": ("1", "1", "1"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--prefix", default="e135_qual")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS),
                    default=list(ARMS))
    ap.add_argument("--telemetry", action="store_true")
    args = ap.parse_args()

    (REPO / "results").mkdir(exist_ok=True)
    for seed in args.seeds:
        order = list(args.arms)
        if seed % 2:
            order.reverse()
        for arm in order:
            out = REPO / "results" / f"{args.prefix}_{arm}_s{seed}.json"
            if out.exists():
                print(f"SKIP {out.relative_to(REPO)}", flush=True)
                continue
            joint, dynamic, body = ARMS[arm]
            env = os.environ.copy()
            env.update({
                "HARVEY_DEVICE": "cpu",
                "HARVEY_WEIGHTS": str(MODEL),
                "HARVEY_JOINT_ROUTES": joint,
                "HARVEY_DYNAMIC_ROUTES": dynamic,
                "HARVEY_BODYBLOCK": body,
            })
            if args.telemetry and arm != "control":
                diag = REPO / "results" / f"{args.prefix}_{arm}_s{seed}.jsonl"
                env["HARVEY_ROUTE_DIAG"] = str(diag)
            cmd = [
                str(PYTHON), "scripts/tournament_eval.py",
                "--agents", "Harvey", "rule_based_agent",
                "rule_based_agent", "rule_based_agent",
                "--n-rounds", str(args.rounds), "--seed", str(seed),
                "--scenario", "classic", "--out", str(out),
                "--match-name", f"{args.prefix}-{arm}-s{seed}",
            ]
            print(f"RUN seed={seed} arm={arm} rounds={args.rounds}", flush=True)
            subprocess.run(cmd, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
