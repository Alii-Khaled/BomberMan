#!/usr/bin/env python3
"""Frozen, evaluation-only Harvey ablation; does not edit tournament code.

The four arms share initial boards and player seats. Per-agent Python/NumPy
module states are isolated; Warden-v2's private generator is not seeded here.
Usage:
  python3 scripts/report_ablation.py battery --seeds 10 --rounds 20 --jobs 8
  python3 scripts/report_ablation.py summarize --seeds 10 --rounds 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "report_ablation"
LOG = ROOT / "logs" / "report_ablation"
ARMS = ("final", "neutral", "search_off", "bc")
FIELDS = {
    "g1": ("rule_based_agent",) * 3,
    "strong": ("warden_v2", "overlord", "sentinel"),
    "archetypes": ("unseen_coward", "unseen_bomber", "unseen_rusher"),
}
FINAL = ROOT / "agent_code" / "Harvey" / "my-saved-model.pt"
BC = ROOT / "results" / "arbiter_ng_d1.pt"


def job_path(arm: str, field: str, seed: int, rounds: int) -> Path:
    return OUT / f"{arm}_{field}_s{seed}_n{rounds}.json"


def seed32(*parts: int) -> int:
    return int(np.random.SeedSequence(parts).generate_state(1)[0])


def run(arm: str, field: str, seed: int, rounds: int, dest: Path) -> None:
    import logging
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import settings as s
    from agents import Agent, AgentRunner
    from environment import BombeRLeWorld, Trophy, WorldArgs

    # World logs consume space and time but supply no metric used here.
    s.LOG_GAME = s.LOG_AGENT_CODE = s.LOG_AGENT_WRAPPER = logging.WARNING
    expected = FINAL if arm != "bc" else BC
    if not expected.is_file():
        raise FileNotFoundError(expected)
    digest = hashlib.sha256(expected.read_bytes()).hexdigest()
    os.environ["HARVEY_WEIGHTS"] = str(expected)
    os.environ["HARVEY_DEVICE"] = "cpu"
    os.environ["HARVEY_SEARCH"] = "off" if arm == "search_off" else "search"
    # No HARVEY_PI_OFF: it skips the expensive forward pass, which would give
    # the neutral arm extra wall-clock search time.
    os.environ.pop("HARVEY_PI_OFF", None)

    original_event = AgentRunner.process_event
    original_wait = Agent.wait_for_act
    decision_times: list[float] = []
    owner_counts = {"search": 0, "other": 0}
    actor_index: dict[str, int] = {}
    actor_rng: dict[tuple[str, int], tuple[object, object]] = {}

    def event(self, event_name, *args):
        if event_name == "act":
            game_state = args[0]
            round_id = int(game_state["round"])
            key = (self.agent_name, round_id)
            if key not in actor_rng:
                rng = seed32(seed, round_id, actor_index[self.agent_name], 27183)
                actor_rng[key] = (random.Random(rng).getstate(),
                                  np.random.RandomState(rng).get_state())
            py_state, np_state = random.getstate(), np.random.get_state()
            try:
                random.setstate(actor_rng[key][0])
                np.random.set_state(actor_rng[key][1])
                result = original_event(self, event_name, *args)
                actor_rng[key] = (random.getstate(), np.random.get_state())
                if self.code_name == "Harvey":
                    owner_counts["search" if getattr(self.fake_self,
                                                       "_search_decided", False)
                                 else "other"] += 1
                return result
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
        result = original_event(self, event_name, *args)
        if event_name == "setup" and self.code_name == "Harvey" and arm == "neutral":
            import torch

            model = getattr(self.fake_self, "_fast", None) or self.fake_self.model
            if model is None:
                raise RuntimeError("Cannot neutralize an absent policy")

            class NeutralPolicy(torch.nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base

                def forward(self, x):
                    logits, value = self.base(x)
                    return torch.zeros_like(logits), value

            # The callback still builds features and runs the full network;
            # search's model/value path remains as in the final arm (V=0).
            self.fake_self._fast = NeutralPolicy(model).eval()
        return result

    def wait(self):
        action, elapsed = original_wait(self)
        if self.code_name == "Harvey":
            decision_times.append(float(elapsed))
        return action, elapsed

    AgentRunner.process_event = event
    Agent.wait_for_act = wait
    LOG.mkdir(parents=True, exist_ok=True)
    log_dir = LOG / f"{arm}_{field}_s{seed}_n{rounds}"
    log_dir.mkdir(parents=True, exist_ok=True)
    args = WorldArgs(no_gui=True, fps=0, turn_based=False,
                     update_interval=0.1, save_replay=False, replay=False,
                     make_video=False, continue_without_training=True,
                     log_dir=str(log_dir), save_stats=False,
                     match_name=None, seed=seed, silence_errors=False,
                     scenario="classic")
    world = BombeRLeWorld(args, [(x, False) for x in ("Harvey", *FIELDS[field])])
    actor_index = {agent.name: i for i, agent in enumerate(world.agents)}
    records = []
    try:
        for round_id in range(1, rounds + 1):
            world.rng = np.random.default_rng(
                np.random.SeedSequence([seed, round_id, 40981]))
            world.new_round()
            board = world.arena.astype(np.int8).tobytes()
            board += json.dumps([(int(coin.x), int(coin.y), bool(coin.collectable))
                                 for coin in world.coins],
                                sort_keys=True).encode()
            board += json.dumps([(a.name, int(a.x), int(a.y))
                                 for a in world.agents]).encode()
            board_hash = hashlib.sha256(board).hexdigest()
            start_t = len(decision_times)
            start_owners = dict(owner_counts)
            while world.running:
                # Separate turn-order RNG from board creation and prior ticks.
                world.rng = np.random.default_rng(np.random.SeedSequence(
                    [seed, round_id, world.step + 1, 15123]))
                world.do_step()
            scores = {a.name: int(a.score) for a in world.agents}
            me = world.agents[0]
            times = decision_times[start_t:]
            records.append({
                "round": round_id, "board_sha256": board_hash,
                "scores": scores,
                "kills": int(me.statistics["kills"]),
                "coins": int(me.statistics["coins"]),
                "suicides": int(me.statistics["suicides"]),
                "bombs": int(me.statistics["bombs"]),
                "invalid": int(me.statistics["invalid"]),
                "timeouts": sum(t is Trophy.time_trophy for t in me.trophies),
                "times_s": times,
                "search_actions": owner_counts["search"] - start_owners["search"],
                "other_actions": owner_counts["other"] - start_owners["other"],
            })
        world.end()
    finally:
        AgentRunner.process_event = original_event
        Agent.wait_for_act = original_wait
    payload = {"arm": arm, "field": field, "seed": seed, "n_rounds": rounds,
               "agents": [a.name for a in world.agents], "weights_sha256": digest,
               "protocol": "report_ablation_v1; independent per-round board/turn RNG; "
                           "separate seeded per-agent Python/NumPy RNG; CPU; "
                           "0.5s official timeout; same current inference code",
               "records": records}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, dest)
    print(f"{arm} {field} seed={seed}: {sum(r['scores']['Harvey'] for r in records)/rounds:.3f}",
          flush=True)


def load(seeds: int, rounds: int):
    data = {}
    for arm in ARMS:
        for field in FIELDS:
            for seed in range(seeds):
                path = job_path(arm, field, seed, rounds)
                d = json.loads(path.read_text())
                if (d["arm"], d["field"], d["seed"], d["n_rounds"]) != (arm, field, seed, rounds):
                    raise ValueError(f"Wrong cohort in {path}")
                if len(d["records"]) != rounds:
                    raise ValueError(f"Missing rounds: {path}")
                data[arm, field, seed] = d
    for field in FIELDS:
        for seed in range(seeds):
            base = [r["board_sha256"] for r in data["final", field, seed]["records"]]
            for arm in ARMS[1:]:
                if base != [r["board_sha256"] for r in data[arm, field, seed]["records"]]:
                    raise ValueError(f"Initial boards differ: {arm}/{field}/{seed}")
    for arm in ARMS:
        hashes = {data[arm, field, seed]["weights_sha256"]
                  for field in FIELDS for seed in range(seeds)}
        if len(hashes) != 1:
            raise ValueError(f"Mixed checkpoint hashes: {arm}: {hashes}")
    return data


def summarize(seeds: int, rounds: int) -> None:
    data = load(seeds, rounds)
    rng = np.random.default_rng(92713)

    def metrics(records):
        scores = np.asarray([r["scores"]["Harvey"] for r in records])
        leaders = [max(r["scores"].values()) for r in records]
        tied = [sum(s == top for s in r["scores"].values())
                for r, top in zip(records, leaders)]
        times = [t for r in records for t in r["times_s"]]
        return {"score": float(scores.mean()),
                "joint_top": float(np.mean(scores == leaders)),
                "fractional_win": float(np.mean([
                    1.0 / n if s == top else 0 for s, n, top in zip(scores, tied, leaders)])),
                **{key: float(np.mean([r[key] for r in records])) for key in
                   ("kills", "coins", "suicides", "bombs", "invalid", "timeouts",
                    "search_actions", "other_actions")},
                "p99_act_s": float(np.percentile(times, 99)) if times else None,
                "max_act_s": max(times) if times else None}

    summary = {"protocol": "report_ablation_v1", "seeds": seeds,
               "rounds_per_seed": rounds, "cohorts": {}, "contrasts": {}}
    for field in FIELDS:
        for arm in ARMS:
            rows = [r for seed in range(seeds) for r in data[arm, field, seed]["records"]]
            summary["cohorts"][f"{arm}/{field}"] = metrics(rows)
        for arm in ARMS[1:]:
            # Resample launch seeds as clusters; the initial states pair
            # within seed/round but the later actions and boards do not.
            blocks = []
            for seed in range(seeds):
                a, b = (data[k, field, seed]["records"] for k in ("final", arm))
                blocks.append(float(np.mean([x["scores"]["Harvey"] - y["scores"]["Harvey"]
                                             for x, y in zip(a, b)])))
            draws = rng.integers(0, seeds, size=(10000, seeds))
            ci = np.percentile(np.asarray(blocks)[draws].mean(axis=1), [2.5, 97.5])
            summary["contrasts"][f"final-minus-{arm}/{field}"] = {
                "score_delta": float(np.mean(blocks)), "seed_block_deltas": blocks,
                "bootstrap_95_seed_blocks": ci.tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"summary_s{seeds}_n{rounds}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("run", "battery", "summarize"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--field", choices=FIELDS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.rounds <= 0 or a.seeds <= 0 or a.jobs <= 0:
        ap.error("rounds, seeds and jobs must be positive")
    if a.mode == "run":
        if a.arm is None or a.field is None or a.seed is None:
            ap.error("run needs --arm, --field and --seed")
        run(a.arm, a.field, a.seed, a.rounds,
            a.out or job_path(a.arm, a.field, a.seed, a.rounds))
    elif a.mode == "summarize":
        summarize(a.seeds, a.rounds)
    else:
        LOG.mkdir(parents=True, exist_ok=True)

        def launch(arm, field, seed):
            dest = job_path(arm, field, seed, a.rounds)
            # A stale/incomplete file is an error rather than silently reused.
            if dest.exists():
                old = json.loads(dest.read_text())
                if (old["arm"], old["field"], old["seed"], old["n_rounds"],
                    len(old["records"])) == (arm, field, seed, a.rounds, a.rounds):
                    return f"SKIP {dest}"
                raise ValueError(f"Existing incompatible result: {dest}")
            cmd = [sys.executable, str(Path(__file__).resolve()), "run",
                   "--arm", arm, "--field", field, "--seed", str(seed),
                   "--rounds", str(a.rounds)]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            (LOG / f"{arm}_{field}_s{seed}_n{a.rounds}.log").write_text(
                proc.stdout + proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"{arm}/{field}/{seed} failed: {proc.stderr[-2000:]}")
            return proc.stdout.strip()

        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            jobs = {ex.submit(launch, arm, field, seed): (arm, field, seed)
                    for seed in range(a.seeds) for field in FIELDS for arm in ARMS}
            for future in as_completed(jobs):
                print(future.result(), flush=True)
        summarize(a.seeds, a.rounds)


if __name__ == "__main__":
    main()
