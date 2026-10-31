"""Extra analyses for the paper, from the saved V2/V2.1 predictions (no GPU needed).

Run from the repository root:  python3 paper/analysis.py
Writes paper/results.json and the figures in paper/figures/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2.1"))
import analyze_v21 as av  # noqa: E402

RES = ROOT / "v2.1" / "results"
OUT = ROOT / "paper"
FIG = OUT / "figures"
LABELS = av.LABELS
RUNS = ["densenet121_seed1", "densenet121_seed2", "efficientnet_b0_seed42"]
N_BOOT = 500
rng = np.random.default_rng(42)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mean_ece(y, p):
    return float(np.mean([av.ece(y[:, k], p[:, k]) for k in range(y.shape[1])]))


def mean_brier(y, p):
    return float(np.mean((p - y) ** 2))


def patient_groups(pids):
    order = np.argsort(pids, kind="stable")
    _, starts = np.unique(pids[order], return_index=True)
    return np.split(order, starts[1:])


def paired_patient_bootstrap(y, a, b, groups, n=N_BOOT):
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(av.mean_auroc(y[idx], a[idx]) - av.mean_auroc(y[idx], b[idx]))
    diffs = np.array(diffs)
    return {"diff": av.mean_auroc(y, a) - av.mean_auroc(y, b),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_le_0": float((diffs <= 0).mean())}


def patient_bootstrap_ci(y, p, groups, n=N_BOOT):
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(av.mean_auroc(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    y_val = np.load(RES / "val_targets.npy").astype(int)
    y = np.load(RES / "test_targets.npy").astype(int)
    val = {r: np.load(RES / f"{r}_val_probs.npy").astype(np.float64) for r in RUNS}
    test = {r: np.load(RES / f"{r}_test_probs.npy").astype(np.float64) for r in RUNS}
    seed42 = np.load(ROOT / "v2" / "results" / "densenet121_test_probs.npy").astype(np.float64)
    with (RES / "test_metadata.csv").open() as f:
        meta = list(csv.DictReader(f))
    pids = np.array([int(m["patient_id"]) for m in meta])
    groups = patient_groups(pids)
    counts = json.loads((OUT / "train_label_counts.json").read_text())
    n_tr = counts["n_train_images"]
    pos = np.array([counts["positives"][l] for l in LABELS], dtype=float)
    w = np.clip((n_tr - pos) / np.maximum(pos, 1), 1, 50)
    out = {"pos_weight": dict(zip(LABELS, w.round(3).tolist()))}

    # 1. Calibration: raw, closed-form logit adjustment (no fitting), Platt fitted on validation.
    cal = {}
    for r in RUNS:
        adj = sigmoid(logit(test[r]) - np.log(w))
        params = av.fit_platt(y_val, val[r])
        platt = av.apply_platt(params, test[r])
        cal[r] = {
            "ece": {"raw": mean_ece(y, test[r]), "logit_adjusted": mean_ece(y, adj), "platt_val": mean_ece(y, platt)},
            "brier": {"raw": mean_brier(y, test[r]), "logit_adjusted": mean_brier(y, adj),
                      "platt_val": mean_brier(y, platt)},
            "platt_slope": [p["a"] for p in params],
            "platt_intercept": [p["b"] for p in params],
            "per_label_ece": {"raw": [av.ece(y[:, k], test[r][:, k]) for k in range(14)],
                              "logit_adjusted": [av.ece(y[:, k], adj[:, k]) for k in range(14)],
                              "platt_val": [av.ece(y[:, k], platt[:, k]) for k in range(14)]},
            "mean_prob": {"raw": test[r].mean(0).tolist(), "logit_adjusted": adj.mean(0).tolist(),
                          "platt_val": platt.mean(0).tolist()},
        }
    out["calibration"] = cal
    b = np.array(cal["densenet121_seed2"]["platt_intercept"])
    out["intercept_vs_log_w"] = {"corr": float(np.corrcoef(b, -np.log(w))[0, 1])}

    # 2. Prevalence shift between validation and test.
    pv, pt = y_val.mean(0), y.mean(0)
    out["prevalence"] = {"val": pv.tolist(), "test": pt.tolist(), "ratio": (pt / pv).tolist(),
                         "any_finding_val": float((y_val.sum(1) > 0).mean()),
                         "any_finding_test": float((y.sum(1) > 0).mean())}
    platt2 = av.apply_platt(av.fit_platt(y_val, val["densenet121_seed2"]), test["densenet121_seed2"])
    signed = platt2.mean(0) - pt  # >0 over-prediction, <0 under-prediction
    out["prevalence"]["signed_gap_after_platt_seed2"] = signed.tolist()
    out["prevalence"]["corr_ratio_vs_signed_gap"] = float(np.corrcoef(np.log(pt / pv), signed / pt)[0, 1])

    # 3. Seeds, ensemble and paired patient-level comparisons.
    dense = {"seed42": seed42, "seed1": test["densenet121_seed1"], "seed2": test["densenet121_seed2"]}
    ens3 = np.mean(list(dense.values()), axis=0)
    ens2_val = (val["densenet121_seed1"] + val["densenet121_seed2"]) / 2
    ens2_test = (test["densenet121_seed1"] + test["densenet121_seed2"]) / 2
    ens2_cal = av.apply_platt(av.fit_platt(y_val, ens2_val), ens2_test)
    out["ensemble"] = {
        "three_seed_mean_auroc": av.mean_auroc(y, ens3),
        "three_seed_ci95": patient_bootstrap_ci(y, ens3, groups),
        "two_seed_calibrated_mean_auroc": av.mean_auroc(y, ens2_cal),
        "two_seed_calibrated_ece": mean_ece(y, ens2_cal),
        "ens3_vs_seed2": paired_patient_bootstrap(y, ens3, dense["seed2"], groups),
        "seed2_vs_efficientnet": paired_patient_bootstrap(y, dense["seed2"], test["efficientnet_b0_seed42"], groups),
        "seed1_vs_seed2": paired_patient_bootstrap(y, dense["seed1"], dense["seed2"], groups),
        "per_label_seed_range": np.ptp(np.array([av.per_class_auroc(y, p) for p in dense.values()]), axis=0).tolist(),
    }

    # 4. Subgroups: how noisy is a per-label AUROC with few positives?
    masks = av.group_masks(meta)
    cells = []
    for g, m in masks.items():
        for k in range(14):
            n_pos = int(y[m, k].sum())
            if n_pos == 0:
                continue
            s = [av.per_class_auroc(y[m][:, [k]], p[m][:, [k]])[0] for p in dense.values()]
            cells.append({"group": g, "label": LABELS[k], "n_pos": n_pos, "seed_range": float(np.ptp(s))})
    out["subgroup_cells"] = cells
    sub = {}
    for g, m in masks.items():
        enough = y[m].sum(0) >= av.MIN_POSITIVES
        per = np.array([av.per_class_auroc(y[m], p[m]) for p in dense.values()])
        sub[g] = {"n_images": int(m.sum()), "all_labels": per.mean(1).tolist(),
                  "enough_positives": per[:, enough].mean(1).tolist(),
                  "excluded": [LABELS[k] for k in np.where(~enough)[0]]}
    out["subgroups_three_dense_seeds"] = sub

    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    figures(out, y, pt)
    print(json.dumps({k: out[k] for k in ["intercept_vs_log_w", "ensemble"]}, indent=2, default=str)[:3000])
    for r in RUNS:
        print(r, {k: round(v, 4) for k, v in cal[r]["ece"].items()}, {k: round(v, 4) for k, v in cal[r]["brier"].items()})
    print("prev", out["prevalence"]["any_finding_val"], out["prevalence"]["any_finding_test"],
          "corr", out["prevalence"]["corr_ratio_vs_signed_gap"])
    for g, v in sub.items():
        print(g, [round(x, 3) for x in v["all_labels"]], [round(x, 3) for x in v["enough_positives"]], v["excluded"])


def figures(out, y, pt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    names = [l.replace("_", " ") for l in LABELS]
    c = out["calibration"]["densenet121_seed2"]
    x = np.arange(14)
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    wd = 0.2
    ax.bar(x - 1.5 * wd, pt, wd, label="Test prevalence", color="#222222")
    ax.bar(x - 0.5 * wd, c["mean_prob"]["raw"], wd, label="Raw", color="#c9a227")
    ax.bar(x + 0.5 * wd, c["mean_prob"]["logit_adjusted"], wd, label="Logit-adjusted (no fitting)", color="#7aa6d6")
    ax.bar(x + 1.5 * wd, c["mean_prob"]["platt_val"], wd, label="Platt (validation)", color="#2e6b3f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("Mean probability")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "calibration.pdf")
    plt.close()

    cells = out["subgroup_cells"]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.scatter([d["n_pos"] for d in cells], [d["seed_range"] for d in cells], s=10, color="#1f4e79", alpha=0.8)
    for d in cells:
        if d["seed_range"] > 0.2:
            ax.annotate(f'{d["label"].replace("_", " ")}, {d["group"]}', (d["n_pos"], d["seed_range"]),
                        fontsize=6, xytext=(4, -2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Positive cases in subgroup (log scale)")
    ax.set_ylabel("AUROC range across 3 seeds")
    plt.tight_layout()
    plt.savefig(FIG / "subgroup_noise.pdf")
    plt.close()


if __name__ == "__main__":
    main()
