"""Reliability analysis of the Lung V2 test predictions (no GPU needed).

Inputs (from the Kaggle run's results/ folder):
  test_targets.npy, densenet121_test_probs.npy, resnet50_test_probs.npy

Outputs (written to analysis/):
  paired_bootstrap.json   DenseNet121 minus ResNet50 mean AUROC, paired bootstrap
  calibration.csv         per-class prevalence, mean predicted probability, ECE, Brier
  recalibration.json      ECE before/after per-class Platt scaling (2-fold cross-fitting on test images)
  selective.csv           mean AUROC on the most confident images at several coverage levels
  calibration_gap.png, selective_auroc.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
SEED = 42


def mean_auroc(y: np.ndarray, p: np.ndarray) -> float:
    scores = [roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]
    return float(np.mean(scores))


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 1000) -> dict:
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(mean_auroc(y[idx], a[idx]) - mean_auroc(y[idx], b[idx]))
    diffs = np.array(diffs)
    return {
        "observed_diff": mean_auroc(y, a) - mean_auroc(y, b),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "share_of_resamples_diff_le_0": float((diffs <= 0).mean()),
        "n_resamples": n,
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def calibration_table(y: np.ndarray, p: np.ndarray) -> list:
    rows = []
    for k, name in enumerate(LABELS):
        rows.append({
            "label": name,
            "prevalence": float(y[:, k].mean()),
            "mean_predicted": float(p[:, k].mean()),
            "ece": ece(y[:, k], p[:, k]),
            "brier": float(np.mean((p[:, k] - y[:, k]) ** 2)),
        })
    return rows


def platt_cross_fit(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-class Platt scaling fitted on one half of the test images and applied to the other half."""
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, 2, len(y))
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    out = np.zeros_like(p)
    for k in range(y.shape[1]):
        for f in (0, 1):
            fit, apply = fold != f, fold == f
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(logit[fit, k:k + 1], y[fit, k])
            out[apply, k] = lr.predict_proba(logit[apply, k:k + 1])[:, 1]
    return out


def selective(y: np.ndarray, p: np.ndarray, coverages: tuple) -> list:
    """Keep the images where the model is most confident (lowest mean binary entropy across labels)."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    entropy = -(q * np.log(q) + (1 - q) * np.log(1 - q)).mean(1)
    order = np.argsort(entropy)
    rows = []
    for c in coverages:
        keep = order[: int(round(c * len(y)))]
        rows.append({"coverage": c, "n_images": int(len(keep)), "mean_auroc": mean_auroc(y[keep], p[keep]),
                     "referred_images": int(len(y) - len(keep))})
    return rows


def write_csv(path: Path, rows: list) -> None:
    keys = list(rows[0].keys())
    lines = [",".join(keys)] + [",".join(str(round(r[k], 4)) if isinstance(r[k], float) else str(r[k]) for k in keys)
                                for r in rows]
    path.write_text("\n".join(lines) + "\n")


def charts(out: Path, calib: list, sel: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["label"] for r in calib]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [r["prevalence"] for r in calib], 0.4, label="Share of test images with the label", color="#1f4e79")
    ax.bar(x + 0.2, [r["mean_predicted"] for r in calib], 0.4, label="Mean predicted probability", color="#c9a227")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("DenseNet121: predicted probability vs actual prevalence (test split)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "calibration_gap.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rows in sel.items():
        ax.plot([r["coverage"] * 100 for r in rows], [r["mean_auroc"] for r in rows], marker="o", label=name)
    ax.set_xlabel("Images kept by the model (%), the rest referred")
    ax.set_ylabel("Mean AUROC on kept images")
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "selective_auroc.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    y = np.load(res / "test_targets.npy").astype(int)
    probs = {name: np.load(res / (name + "_test_probs.npy")).astype(np.float64) for name in ["densenet121", "resnet50"]}

    pb = paired_bootstrap(y, probs["densenet121"], probs["resnet50"])
    (out / "paired_bootstrap.json").write_text(json.dumps(pb, indent=2))

    calib = calibration_table(y, probs["densenet121"])
    write_csv(out / "calibration.csv", calib)

    recal = platt_cross_fit(y, probs["densenet121"])
    rec = {
        "mean_ece_before": float(np.mean([r["ece"] for r in calib])),
        "mean_ece_after": float(np.mean([ece(y[:, k], recal[:, k]) for k in range(14)])),
        "mean_brier_before": float(np.mean([r["brier"] for r in calib])),
        "mean_brier_after": float(np.mean((recal - y) ** 2)),
        "mean_auroc_after": mean_auroc(y, recal),
        "note": "2-fold cross-fitting on random halves of the test images; patient IDs were not used for the halves",
    }
    (out / "recalibration.json").write_text(json.dumps(rec, indent=2))

    coverages = (1.0, 0.9, 0.8, 0.7, 0.5)
    sel = {name: selective(y, p, coverages) for name, p in probs.items()}
    write_csv(out / "selective.csv", [dict(model=n, **r) for n, rows in sel.items() for r in rows])
    charts(out, calib, sel)
    print(json.dumps({"paired_bootstrap": pb, "recalibration": rec}, indent=2))
    for r in calib:
        print(r)
    for n, rows in sel.items():
        for r in rows:
            print(n, r)


if __name__ == "__main__":
    main()
