#!/usr/bin/env python3
"""Bundle the original 317 report inputs into a local archive.

The public raw evidence and its manifest are checked separately by
verify_report_evidence.py. This legacy archive excludes the additional
appendix wildcard matches and Warden-v1/v2 comparison files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent
FIGURE_MANIFEST = ROOT / "evidence" / "figure_values.json"
ARCHIVE = ROOT / "evidence" / "evaluation-records.tar.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify inputs against saved hashes")
    a = ap.parse_args()
    figure = json.loads(FIGURE_MANIFEST.read_text())
    expected = figure["sources_sha256"]
    if len(expected) != 193:
        raise ValueError(f"Unexpected historical source count: {len(expected)}")
    for rel, digest in expected.items():
        path = ROOT / rel
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source hash changed: {rel}")
    extra = [ROOT / "logs" / "e108_bc.log",
             ROOT / "results" / "arbiter_ng_d1.meta.json",
             ROOT / "results" / "arbiter_ng_d1.pt",
             ROOT / "results" / "report_ablation" / "summary_s10_n20.json"]
    ablation_dir = ROOT / "results" / "report_ablation"
    ablations = [ablation_dir / f"{arm}_{field}_s{seed}_n20.json"
                 for arm in ("final", "neutral", "search_off", "bc")
                 for field in ("g1", "strong", "archetypes")
                 for seed in range(10)]
    if not all(path.is_file() for path in ablations):
        raise ValueError("One or more of the 120 expected ablation cohorts are missing")
    summary = json.loads(extra[-1].read_text())
    if summary["seeds"] != 10 or summary["rounds_per_seed"] != 20 or len(summary["contrasts"]) != 9:
        raise ValueError("Wrong report-ablation summary")
    extra.extend(ablations)
    items = sorted({*(ROOT / p for p in expected), *extra}, key=str)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in items}
    code = [ROOT / path for path in (
        "scripts/report_ablation.py", "agents.py", "environment.py",
        "settings.py", "fallbacks.py", "items.py",
        "agent_code/Harvey/my-saved-model.pt")]
    for directory in ("Harvey", "rule_based_agent", "overlord", "sentinel"):
        code.extend((ROOT / "agent_code" / directory).glob("*.py"))
    for directory in ("warden_v2", "unseen_coward", "unseen_bomber", "unseen_rusher"):
        code.extend((ROOT / "outsiders" / directory).glob("*.py"))
    code_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(code)}
    if len(code_hashes) < 30:
        raise ValueError("Missing evaluation code or opponents")
    for arm in ("final", "neutral", "search_off", "bc"):
        checkpoint = (ROOT / "results" / "arbiter_ng_d1.pt" if arm == "bc"
                      else ROOT / "agent_code" / "Harvey" / "my-saved-model.pt")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for path in (p for p in ablations if p.name.startswith(f"{arm}_")):
            if json.loads(path.read_text())["weights_sha256"] != digest:
                raise ValueError(f"Cohort used different {arm} weights: {path}")
    manifest = {"note": "Evaluation sources and D1 BC weights; no report, PDF, demo corpora or optimizer states.",
                "files": sources, "figure_inputs": len(expected),
                "ablation_cohorts": len(ablations), "code_sha256": code_hashes}
    if a.check:
        if not ARCHIVE.is_file():
            raise FileNotFoundError(ARCHIVE)
        with tarfile.open(ARCHIVE, "r:gz") as tf:
            members = tf.getmembers()
            names = [member.name for member in members]
            expected_names = set(sources) | {"manifest.json"}
            if (any(not member.isfile() for member in members)
                    or set(names) != expected_names
                    or len(names) != len(expected_names)):
                raise ValueError("Unexpected files in evidence archive")
            recorded = json.load(tf.extractfile("manifest.json"))
            if recorded != manifest:
                raise ValueError("Evidence archive manifest is stale")
            for name, digest in sources.items():
                stream = tf.extractfile(name)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                    raise ValueError(f"Archive hash mismatch: {name}")
        print(f"PASS: {len(sources)} evidence sources match archive")
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as tf:
        for path in items:
            rel = str(path.relative_to(ROOT))
            raw = path.read_bytes()
            info = tarfile.TarInfo(rel)
            info.size = len(raw)
            info.mtime = 0
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(raw))
        body = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(body)
        info.mtime = 0
        info.mode = 0o644
        tf.addfile(info, io.BytesIO(body))
    print(f"Packed {len(sources)} evaluation inputs into {ARCHIVE}")


if __name__ == "__main__":
    main()
