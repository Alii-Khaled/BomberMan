#!/usr/bin/env python3
"""Verify all published appendix inputs without access to the private report.

Run from a fresh clone with: python3 -B scripts/verify_report_evidence.py
The public manifest is an integrity record, not a substitute for the
separate evaluation protocol in scripts/report_ablation.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

from check_equivalence import compare


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "manifest.json"
REVIEW = ROOT / "evidence" / "source_review.json"


def safe_path(rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != rel:
        raise ValueError(f"Unsafe manifest path: {rel}")
    return ROOT / path


def check_files(fingerprints: dict[str, str], label: str) -> None:
    for rel, expected in fingerprints.items():
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid SHA-256 for {rel}")
        actual = hashlib.sha256(safe_path(rel).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{label} hash mismatch: {rel}")
    print(f"PASS: {len(fingerprints)} {label} files match SHA-256")


def check_evaluation_code(fingerprints: dict[str, str]) -> None:
    """Verify original sources and the explicitly recorded review edits."""
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    if review["version"] != 1:
        raise ValueError("Unsupported source-review version")
    revision = review["baseline_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid baseline commit")
    reviewed = review["files"]
    if set(reviewed) - set(fingerprints):
        raise ValueError("Source review includes unindexed files")
    unchanged = {rel: sha for rel, sha in fingerprints.items() if rel not in reviewed}
    check_files(unchanged, "unchanged evaluation code/checkpoint")
    check_files({rel: entry["sha256"] for rel, entry in reviewed.items()},
                "reviewed source")
    for rel, entry in reviewed.items():
        path = safe_path(rel)
        if path.suffix != ".py":
            raise ValueError(f"Only Python source can have review edits: {rel}")
        try:
            original = subprocess.check_output(
                ["git", "show", f"{revision}:{rel}"], cwd=ROOT,
                stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Source provenance requires Git commit {revision}. "
                "Use a full clone, or fetch the missing history.") from exc
        if hashlib.sha256(original).hexdigest() != fingerprints[rel]:
            raise ValueError(f"Original evaluation-code hash mismatch: {rel}")
        source = original.decode("utf-8")
        # A recorded output-path change is separate from comment-only edits.
        for old, new in entry.get("source_replacements", []):
            if not old or source.count(old) != 1:
                raise ValueError(f"Ambiguous reviewed replacement: {rel}")
            source = source.replace(old, new, 1)
        problems = compare(source, path.read_text(encoding="utf-8"))
        if problems:
            raise ValueError(f"Unrecorded executable change: {rel}: {problems}")
    print(f"PASS: {len(reviewed)} reviewed sources match the original code "
          "apart from comments/docstrings and recorded path changes")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["version"] != 1:
        raise ValueError("Unsupported evidence manifest version")
    sources = manifest["sources_sha256"]
    if len(sources) != manifest["source_count"] or len(sources) != 489:
        raise ValueError("Wrong number of published result/log/checkpoint files")
    check_files(sources, "appendix source")
    check_files(manifest["derived_sha256"], "figure-input fingerprint")
    check_evaluation_code(manifest["evaluation_code_sha256"])

    for name in ("figure_values.json", "training_values.json", "ablation_values.json"):
        values = json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))
        for rel, expected in values["sources_sha256"].items():
            if sources.get(rel) != expected:
                raise ValueError(f"{name} disagrees with the public manifest: {rel}")
        print(f"PASS: {name} indexes {len(values['sources_sha256'])} published inputs")


if __name__ == "__main__":
    main()
