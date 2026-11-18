"""Patient-level interval and calibration (raw and logit-adjusted) for DenseNet121 seed 42.

Seed 42 was trained in v2, which saved test but not validation probabilities, so Platt scaling
cannot be fitted for it. Run from the repository root: python3 paper/seed42_check.py
"""
import csv
import json

import numpy as np

import analysis as an

y = np.load(an.RES / "test_targets.npy").astype(int)
p = np.load(an.ROOT / "v2" / "results" / "densenet121_test_probs.npy").astype(np.float64)
with (an.RES / "test_metadata.csv").open() as f:
    pids = np.array([int(m["patient_id"]) for m in csv.DictReader(f)])
groups = an.patient_groups(pids)
counts = json.loads((an.OUT / "train_label_counts.json").read_text())
pos = np.array([counts["positives"][l] for l in an.LABELS], dtype=float)
w = np.clip((counts["n_train_images"] - pos) / np.maximum(pos, 1), 1, 50)
adj = an.sigmoid(an.logit(p) - np.log(w))
an.rng = np.random.default_rng(42)
out = {
    "mean_auroc": an.av.mean_auroc(y, p),
    "patient_ci95": an.patient_bootstrap_ci(y, p, groups),
    "ece": {"raw": an.mean_ece(y, p), "logit_adjusted": an.mean_ece(y, adj)},
    "brier": {"raw": an.mean_brier(y, p), "logit_adjusted": an.mean_brier(y, adj)},
}
(an.OUT / "seed42.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
