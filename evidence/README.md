# Report evaluation records

`manifest.json` indexes **489 tracked raw files** under `results/` and
`logs/` by SHA-256: the historical figure inputs, the report-stage
ablation cohorts, the D1 cloning checkpoint and log, all surviving local
matches for the appendix's result patterns, and the two E94 Warden-v1/v2
head-to-head JSON files. The raw files retain the paths printed in the
appendix. Only the files named in the manifest were published; other
training output in `results/` remains ignored by Git.

The figure-input fingerprints live here as `figure_values.json`,
`training_values.json` and `ablation_values.json`. Their copies inside the
private report source are identical. `manifest.json` also records hashes
of the evaluation code and opponent source used to check the final ablation.

From a fresh clone, run:

```bash
python3 -B scripts/verify_report_evidence.py
python3 -B scripts/audit_report_results.py
```

The first command checks every published input and source fingerprint;
the second recomputes the E130 identity check and nine final-ablation
score intervals without playing new matches. Re-running the battery
requires PyTorch, NumPy, the included agent weights and CPU time.

The earlier **317-file** `evaluation-records.tar.gz` is a local subset
of the published raw files. It is not tracked and is not needed for the
public checks. If you have that local archive, `python3 -B
scripts/package_report_evidence.py --check` verifies it without using
the private report directory. The E108 selected RL weights have the same
SHA-256 as `agent_code/Harvey/my-saved-model.pt` and also appear at the
appendix's `results/e108rl.pt.ep225` path.

The private report PDF and source, original demonstration corpus, and
some early ledger-only result files are not in this repository. The
public data reproduce recorded aggregates, not an unseen tournament
ranking or the historical training run from scratch.
