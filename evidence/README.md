# Report evaluation records

`manifest.json` indexes **489 tracked raw files** under `results/` and
`logs/` by SHA-256: the historical figure inputs, the report-stage
ablation cohorts, the D1 cloning checkpoint and log, all surviving local
matches for the appendix's result patterns, and the two E94 Warden-v1/v2
head-to-head JSON files. The raw files retain the paths printed in the
appendix. Only the files named in the manifest were published; other
training output in `results/` remains ignored by Git.

The figure-input fingerprints are `figure_values.json`,
`training_values.json` and `ablation_values.json`. `manifest.json` also
records hashes of the evaluation code and opponent source used to check
the final ablation.

`source_review.json` records the later documentation cleanup against Git
commit `5ef5dc67d71a1c6c9140c640794b7b6799d173ae`. The original manifest's
hashes remain intact. The verifier checks the original bytes from that
commit, the reviewed files' hashes, and their executable tokens and syntax
trees. Agent changes consist only of comments/docstrings. The ablation
runner also redirects logs to the repository's `logs/report_ablation/`;
the review records that source replacement explicitly. Use a full Git
clone for this source-history check.

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
