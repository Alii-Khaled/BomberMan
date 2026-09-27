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


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "manifest.json"


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


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["version"] != 1:
        raise ValueError("Unsupported evidence manifest version")
    sources = manifest["sources_sha256"]
    if len(sources) != manifest["source_count"] or len(sources) != 489:
        raise ValueError("Wrong number of published result/log/checkpoint files")
    check_files(sources, "appendix source")
    check_files(manifest["derived_sha256"], "figure-input fingerprint")
    check_files(manifest["evaluation_code_sha256"], "evaluation code/checkpoint")

    for name in ("figure_values.json", "training_values.json", "ablation_values.json"):
        values = json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))
        for rel, expected in values["sources_sha256"].items():
            if sources.get(rel) != expected:
                raise ValueError(f"{name} disagrees with the public manifest: {rel}")
        print(f"PASS: {name} indexes {len(values['sources_sha256'])} published inputs")


if __name__ == "__main__":
    main()
