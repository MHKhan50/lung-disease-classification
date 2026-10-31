"""Analysis of the Lung V2.1 predictions (no GPU needed).

Re-reads the probabilities saved by the Kaggle run and writes every table and chart used in the README.

Inputs (results/):
  val_targets.npy, test_targets.npy
  <run>_val_probs.npy, <run>_test_probs.npy for densenet121_seed1, densenet121_seed2, efficientnet_b0_seed42
  test_metadata.csv (patient ID, age, sex and view of each test image, in the same order as test_targets.npy)
  ../v2/results/densenet121_test_probs.npy (the V2 DenseNet121 run, seed 42), used for the seed comparison

Outputs (analysis/):
  seeds.json              test mean AUROC of the three DenseNet121 seeds and of their averaged prediction
  per_class_auroc.csv     per-class test AUROC for every run
  calibration.csv         per-class prevalence, mean probability, ECE and Brier before and after calibration
  calibration_summary.json
  platt_params.json       per-class Platt parameters fitted on the validation set (DenseNet121 seed 2)
  referral.csv            mean AUROC on the most confident images at several coverage levels
  subgroups.csv           mean AUROC by sex and age band, over all labels and over labels with enough positives
  subgroup_positives.csv  positive cases per label in each subgroup
  per_class_auroc.png, calibration.png, referral.png, subgroups.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
RUNS = ["densenet121_seed1", "densenet121_seed2", "efficientnet_b0_seed42"]
APP_RUN = "densenet121_seed2"  # highest validation mean AUROC (0.831); chosen on validation, not test
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
MIN_POSITIVES = 10  # labels with fewer positives in a subgroup give an AUROC too noisy to average
GROUPS = ["sex=M", "sex=F", "age<40", "age 40-59", "age 60-100"]


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> list:
    return [float(roc_auc_score(y[:, k], p[:, k])) if 0 < y[:, k].sum() < len(y) else float("nan")
            for k in range(y.shape[1])]


def mean_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean(per_class_auroc(y, p)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(y_fit: np.ndarray, p_fit: np.ndarray) -> list:
    """Per-class Platt scaling: calibrated = sigmoid(a * logit(p) + b), fitted on held-out data."""
    params = []
    for k in range(y_fit.shape[1]):
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(logit(p_fit[:, k:k + 1]), y_fit[:, k])
        params.append({"label": LABELS[k], "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})
    return params


def apply_platt(params: list, p: np.ndarray) -> np.ndarray:
    a = np.array([q["a"] for q in params])
    b = np.array([q["b"] for q in params])
    return 1 / (1 + np.exp(-(a * logit(p) + b)))


def confidence_order(p: np.ndarray) -> np.ndarray:
    """Image indices from most to least confident (lowest to highest mean binary entropy across labels)."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    entropy = -(q * np.log(q) + (1 - q) * np.log(1 - q)).mean(1)
    return np.argsort(entropy, kind="stable")


def referral(y: np.ndarray, p: np.ndarray, coverages: tuple = COVERAGES) -> list:
    """Keep the most confident share of images, refer the rest, and score the kept images."""
    order = confidence_order(p)
    rows = []
    for c in coverages:
        keep = order[: int(round(c * len(y)))]
        rows.append({"coverage": c, "kept_images": int(len(keep)), "referred_images": int(len(y) - len(keep)),
                     "mean_auroc_kept": mean_auroc(y[keep], p[keep])})
    return rows


def group_masks(meta: list) -> dict:
    age = np.array([int(m["age"]) for m in meta])
    sex = np.array([m["sex"] for m in meta])
    return {"sex=M": sex == "M", "sex=F": sex == "F", "age<40": age < 40,
            "age 40-59": (age >= 40) & (age < 60), "age 60-100": (age >= 60) & (age <= 100)}


def subgroup_rows(y: np.ndarray, p: np.ndarray, masks: dict, min_pos: int = MIN_POSITIVES) -> list:
    """Mean AUROC per subgroup over all labels, and over labels with at least min_pos positives there."""
    rows = []
    for g, m in masks.items():
        scores = np.array(per_class_auroc(y[m], p[m]))
        enough = y[m].sum(0) >= min_pos
        rows.append({"group": g, "n_images": int(m.sum()), "mean_auroc_all_labels": float(np.nanmean(scores)),
                     "mean_auroc_labels_with_enough_positives": float(np.nanmean(scores[enough])),
                     "labels_excluded": ";".join(LABELS[k] for k in np.where(~enough)[0])})
    return rows


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def charts(out: Path, auroc: dict, calib: list, ref: dict, subgroups: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"DenseNet121 seed 42 (V2)": "#9db4cf", "DenseNet121 seed 1": "#1f4e79",
              "DenseNet121 seed 2": "#3d7ab8", "EfficientNet-B0": "#c9a227"}
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.8 / len(auroc)
    for i, (name, scores) in enumerate(auroc.items()):
        ax.bar(x + (i - (len(auroc) - 1) / 2) * w, scores, w, label=name, color=colors.get(name))
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_ylim(0.6, 0.95)
    ax.set_ylabel("Test AUROC")
    ax.set_title("Per-class test AUROC, official patient-wise split")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out / "per_class_auroc.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.27
    ax.bar(x - w, [r["prevalence"] for r in calib], w, label="Share of test images with the label", color="#1f4e79")
    ax.bar(x, [r["mean_prob_raw"] for r in calib], w, label="Mean probability, raw", color="#c9a227")
    ax.bar(x + w, [r["mean_prob_calibrated"] for r in calib], w, label="Mean probability, calibrated on validation",
           color="#5a9e6f")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("DenseNet121 seed 2: raw vs validation-calibrated probabilities (test split)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "calibration.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, rows in ref.items():
        ax.plot([r["coverage"] * 100 for r in rows], [r["mean_auroc_kept"] for r in rows], marker="o", label=name)
    ax.set_xlabel("Images kept (%), the rest referred to a reader")
    ax.set_ylabel("Mean AUROC on kept images")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "referral.png", dpi=150)
    plt.close()

    groups = GROUPS
    runs = list(dict.fromkeys(r["run"] for r in subgroups))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    gx = np.arange(len(groups))
    w = 0.8 / len(runs)
    names = {"densenet121_seed1": "DenseNet121 seed 1", "densenet121_seed2": "DenseNet121 seed 2",
             "efficientnet_b0_seed42": "EfficientNet-B0"}
    for i, run in enumerate(runs):
        vals = [next(r["mean_auroc_labels_with_enough_positives"] for r in subgroups if r["run"] == run and r["group"] == g)
                for g in groups]
        ax.bar(gx + (i - (len(runs) - 1) / 2) * w, vals, w, label=names.get(run, run), color=colors.get(names.get(run)))
    ax.set_xticks(gx)
    ax.set_xticklabels(groups)
    ax.set_ylim(0.70, 0.82)
    ax.set_ylabel("Mean test AUROC")
    ax.set_title(f"Mean AUROC by sex and age band (labels with at least {MIN_POSITIVES} positives in the group)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "subgroups.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--v2-results", default="../v2/results")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    y_val = np.load(res / "val_targets.npy").astype(int)
    y_test = np.load(res / "test_targets.npy").astype(int)
    val = {r: np.load(res / (r + "_val_probs.npy")).astype(np.float64) for r in RUNS}
    test = {r: np.load(res / (r + "_test_probs.npy")).astype(np.float64) for r in RUNS}
    v2_test = np.load(Path(args.v2_results) / "densenet121_test_probs.npy").astype(np.float64)
    if not np.array_equal(np.load(Path(args.v2_results) / "test_targets.npy").astype(int), y_test):
        raise SystemExit("V2 and V2.1 test targets differ; the runs are not comparable")

    # Seeds: three DenseNet121 runs on the same split, and their averaged prediction.
    dense = {"seed 42 (V2)": v2_test, "seed 1": test["densenet121_seed1"], "seed 2": test["densenet121_seed2"]}
    seed_scores = {k: mean_auroc(y_test, p) for k, p in dense.items()}
    vals = np.array(list(seed_scores.values()))
    seeds = {
        "test_mean_auroc": seed_scores,
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "range": float(vals.max() - vals.min()),
        "three_seed_average_prediction_test_mean_auroc": mean_auroc(y_test, np.mean(list(dense.values()), axis=0)),
    }
    (out / "seeds.json").write_text(json.dumps(seeds, indent=2))

    auroc = {"DenseNet121 seed 42 (V2)": per_class_auroc(y_test, v2_test),
             "DenseNet121 seed 1": per_class_auroc(y_test, test["densenet121_seed1"]),
             "DenseNet121 seed 2": per_class_auroc(y_test, test["densenet121_seed2"]),
             "EfficientNet-B0": per_class_auroc(y_test, test["efficientnet_b0_seed42"])}
    write_csv(out / "per_class_auroc.csv",
              [dict(label=l, **{name: s[k] for name, s in auroc.items()}) for k, l in enumerate(LABELS)])

    # Calibration fitted on validation, checked on test.
    summary, calibrated = {}, {}
    for r in RUNS:
        params = fit_platt(y_val, val[r])
        cal = apply_platt(params, test[r])
        calibrated[r] = cal
        summary[r] = {
            "test_mean_auroc": mean_auroc(y_test, test[r]),
            "test_mean_auroc_calibrated": mean_auroc(y_test, cal),
            "mean_ece_raw": float(np.mean([ece(y_test[:, k], test[r][:, k]) for k in range(14)])),
            "mean_ece_calibrated": float(np.mean([ece(y_test[:, k], cal[:, k]) for k in range(14)])),
            "mean_brier_raw": float(np.mean((test[r] - y_test) ** 2)),
            "mean_brier_calibrated": float(np.mean((cal - y_test) ** 2)),
        }
        if r == APP_RUN:
            (out / "platt_params.json").write_text(json.dumps(
                {"run": r, "fitted_on": "validation split (8,805 images, patient-wise)", "labels": params}, indent=2))
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2))

    raw, cal = test[APP_RUN], calibrated[APP_RUN]
    calib = [{"label": l, "prevalence": float(y_test[:, k].mean()),
              "mean_prob_raw": float(raw[:, k].mean()), "mean_prob_calibrated": float(cal[:, k].mean()),
              "ece_raw": ece(y_test[:, k], raw[:, k]), "ece_calibrated": ece(y_test[:, k], cal[:, k]),
              "brier_raw": float(np.mean((raw[:, k] - y_test[:, k]) ** 2)),
              "brier_calibrated": float(np.mean((cal[:, k] - y_test[:, k]) ** 2))}
             for k, l in enumerate(LABELS)]
    write_csv(out / "calibration.csv", calib)

    # Referral: keep the most confident images (fixes the reversed order in the Kaggle notebook).
    names = {"densenet121_seed1": "DenseNet121 seed 1", "densenet121_seed2": "DenseNet121 seed 2",
             "efficientnet_b0_seed42": "EfficientNet-B0"}
    ref, ref_rows = {}, []
    for r in RUNS:
        for kind, p in (("raw", test[r]), ("calibrated", calibrated[r])):
            rows = referral(y_test, p)
            ref_rows += [dict(run=r, probabilities=kind, **row) for row in rows]
            if kind == "calibrated":
                ref[names[r]] = rows
    write_csv(out / "referral.csv", ref_rows)

    with (res / "test_metadata.csv").open() as f:
        meta = list(csv.DictReader(f))
    if len(meta) != len(y_test):
        raise SystemExit("test_metadata.csv does not match test_targets.npy")
    masks = group_masks(meta)
    subgroups = [dict(run=r, **row) for r in RUNS for row in subgroup_rows(y_test, test[r], masks)]
    write_csv(out / "subgroups.csv", subgroups)
    write_csv(out / "subgroup_positives.csv",
              [dict(label=l, **{g: int(y_test[m, k].sum()) for g, m in masks.items()}) for k, l in enumerate(LABELS)])
    charts(out, auroc, calib, ref, subgroups)

    print(json.dumps({"seeds": seeds, "calibration": summary}, indent=2))
    for row in ref_rows:
        print(row)
    for row in subgroups:
        print(row)


if __name__ == "__main__":
    main()
