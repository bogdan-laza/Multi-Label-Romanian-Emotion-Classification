"""Compare Method 1 (SVM) vs Method 2 (MLP) on the test set; export tables and plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    COMPARISON_RESULTS_DIR,
    DEDUPLICATE_TEXTS,
    LABEL_NAMES,
    MLP_FEATURES_DIR,
    MLP_MODEL_DIR,
    MLP_RESULTS_DIR,
    MODELS_DIR,
    SVM_MODEL_DIR,
    SVM_RESULTS_DIR,
)
from src.data import get_X_y, load_all_splits  # noqa: E402
from src.evaluate import binarize_proba, compute_multilabel_metrics  # noqa: E402
from src.features import TfidfFeaturePipeline  # noqa: E402
from src.train_svm import decision_scores, load_svm_artifacts, predict_multilabel  # noqa: E402

import joblib  # noqa: E402


def load_metrics_files() -> tuple[dict, dict]:
    svm_path = SVM_RESULTS_DIR / "report.json"
    mlp_path = MLP_RESULTS_DIR / "mlp_metrics.json"
    if not mlp_path.is_file():
        mlp_path = ROOT / "results" / "mlp_metrics.json"
    if not svm_path.is_file():
        raise FileNotFoundError(f"Missing SVM results at {svm_path}. Run scripts/train_svm.py")
    if not mlp_path.is_file():
        raise FileNotFoundError(f"Missing MLP results at {mlp_path}. Run scripts/train_mlp.py")
    with svm_path.open(encoding="utf-8") as f:
        svm_report = json.load(f)
    with mlp_path.open(encoding="utf-8") as f:
        mlp_report = json.load(f)
    return svm_report, mlp_report


def evaluate_on_test():
    splits = load_all_splits(deduplicate=DEDUPLICATE_TEXTS)
    _, y_test = get_X_y(splits["test"])
    X_test, _ = get_X_y(splits["test"])

    svm_model, svm_thresholds, _ = load_svm_artifacts(SVM_MODEL_DIR)
    svm_pipe = TfidfFeaturePipeline.load(MODELS_DIR / "tfidf")
    X_te_svm = svm_pipe.transform(X_test)
    y_svm = predict_multilabel(svm_model, X_te_svm, thresholds=svm_thresholds)
    svm_scores = decision_scores(svm_model, X_te_svm)

    mlp = joblib.load(MLP_MODEL_DIR / "mlp_classifier.joblib")
    mlp_thresholds = np.load(MLP_MODEL_DIR / "thresholds.npy")
    mlp_pipe = TfidfFeaturePipeline.load(MLP_FEATURES_DIR)
    X_te_mlp = mlp_pipe.transform(X_test, apply_svd=True)
    mlp_proba = mlp.predict_proba(X_te_mlp)
    y_mlp = binarize_proba(mlp_proba, mlp_thresholds)

    svm_metrics = compute_multilabel_metrics(y_test, y_svm)
    mlp_metrics = compute_multilabel_metrics(y_test, y_mlp)
    return (
        svm_metrics,
        mlp_metrics,
        y_test,
        y_svm,
        y_mlp,
        svm_scores,
        mlp_proba,
        svm_thresholds,
        mlp_thresholds,
        X_test,
    )


def main() -> None:
    COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    svm_report, mlp_report = load_metrics_files()
    (
        svm_metrics,
        mlp_metrics,
        y_test,
        y_svm,
        y_mlp,
        svm_scores,
        mlp_proba,
        svm_thresholds,
        mlp_thresholds,
        texts,
    ) = evaluate_on_test()

    summary = pd.DataFrame(
        [
            {
                "model": "SVM (Binary Relevance + LinearSVC)",
                "macro_f1": svm_metrics["macro_f1"],
                "micro_f1": svm_metrics["micro_f1"],
                "hamming_loss": svm_metrics["hamming_loss"],
                "subset_accuracy": svm_metrics["subset_accuracy"],
            },
            {
                "model": "MLP (TF-IDF + SVD)",
                "macro_f1": mlp_metrics["macro_f1"],
                "micro_f1": mlp_metrics["micro_f1"],
                "hamming_loss": mlp_metrics["hamming_loss"],
                "subset_accuracy": mlp_metrics["subset_accuracy"],
            },
        ]
    )
    summary.to_csv(COMPARISON_RESULTS_DIR / "test_metrics_comparison.csv", index=False)

    per_label = svm_metrics["per_label"].rename(
        columns=lambda c: f"svm_{c}" if c != "label" else c
    )
    per_label = per_label.merge(
        mlp_metrics["per_label"].rename(
            columns=lambda c: f"mlp_{c}" if c != "label" else c
        ),
        on="label",
    )
    per_label["f1_delta_mlp_minus_svm"] = per_label["mlp_f1"] - per_label["svm_f1"]
    per_label.to_csv(COMPARISON_RESULTS_DIR / "per_label_comparison.csv", index=False)

    disagree = np.any(y_svm != y_mlp, axis=1)
    disagree_df = pd.DataFrame(
        {
            "test_index": np.where(disagree)[0],
            "text": [texts[i] for i in np.where(disagree)[0]],
            "true_labels": [
                ", ".join(LABEL_NAMES[j] for j in range(len(LABEL_NAMES)) if y_test[i, j])
                for i in np.where(disagree)[0]
            ],
            "svm_pred": [
                ", ".join(LABEL_NAMES[j] for j in range(len(LABEL_NAMES)) if y_svm[i, j])
                for i in np.where(disagree)[0]
            ],
            "mlp_pred": [
                ", ".join(LABEL_NAMES[j] for j in range(len(LABEL_NAMES)) if y_mlp[i, j])
                for i in np.where(disagree)[0]
            ],
        }
    )
    disagree_df.to_csv(COMPARISON_RESULTS_DIR / "disagreements.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics_plot = ["macro_f1", "micro_f1", "subset_accuracy"]
    x = np.arange(len(metrics_plot))
    w = 0.35
    axes[0].bar(x - w / 2, [summary.iloc[0][m] for m in metrics_plot], w, label="SVM")
    axes[0].bar(x + w / 2, [summary.iloc[1][m] for m in metrics_plot], w, label="MLP")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_plot)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Test set: overall metrics")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].bar(per_label["label"], per_label["svm_f1"], width=0.35, label="SVM F1")
    axes[1].bar(
        np.arange(len(per_label)) + 0.35,
        per_label["mlp_f1"],
        width=0.35,
        label="MLP F1",
    )
    axes[1].set_xticks(np.arange(len(per_label)) + 0.175)
    axes[1].set_xticklabels(per_label["label"], rotation=45, ha="right")
    axes[1].set_ylabel("F1")
    axes[1].set_title("Per-label F1 (test)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(COMPARISON_RESULTS_DIR / "svm_vs_mlp_comparison.png", dpi=150)
    plt.close()

    report = {
        "svm_test_from_report": svm_report.get("test", {}),
        "mlp_test_from_report": mlp_report.get("test", {}),
        "svm_test_recomputed": {
            k: float(v) for k, v in svm_metrics.items() if k != "per_label"
        },
        "mlp_test_recomputed": {
            k: float(v) for k, v in mlp_metrics.items() if k != "per_label"
        },
        "n_test_disagreements": int(disagree.sum()),
        "n_test_samples": len(texts),
    }
    with (COMPARISON_RESULTS_DIR / "comparison_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== SVM vs MLP (test set) ===")
    print(summary.to_string(index=False))
    print(f"\nDisagreements: {disagree.sum()} / {len(texts)} tweets")
    print(f"\nSaved to {COMPARISON_RESULTS_DIR}")


if __name__ == "__main__":
    main()
