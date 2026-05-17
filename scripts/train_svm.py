from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEDUPLICATE_TEXTS,
    LABEL_NAMES,
    MODELS_DIR,
    RANDOM_SEED,
    SVM_MODEL_DIR,
    SVM_RESULTS_DIR,
)
from src.data import get_X_y, load_all_splits  # noqa: E402
from src.evaluate import compute_multilabel_metrics  # noqa: E402
from src.features import TfidfFeaturePipeline  # noqa: E402
from src.train_svm import (  # noqa: E402
    evaluate_baseline,
    predict_multilabel,
    save_svm_artifacts,
    top_tfidf_coefficients,
    train_and_tune,
)


def _load_or_build_tfidf(X_train, X_valid, X_test):
    tfidf_dir = MODELS_DIR / "tfidf"
    if tfidf_dir.is_dir() and (tfidf_dir / "vectorizer.joblib").is_file():
        pipe = TfidfFeaturePipeline.load(tfidf_dir)
        print(f"Loaded TF-IDF pipeline from {tfidf_dir}")
    else:
        pipe = TfidfFeaturePipeline(use_svd=False)
        pipe.fit(X_train)
        pipe.save(tfidf_dir)
        print(f"Fitted and saved TF-IDF pipeline to {tfidf_dir}")

    return (
        pipe.transform(X_train),
        pipe.transform(X_valid),
        pipe.transform(X_test),
        pipe,
    )


def _metrics_to_float(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if k == "per_label":
            continue
        out[k] = float(v) if isinstance(v, (float, np.floating)) else v
    return out


def main() -> None:
    splits = load_all_splits(deduplicate=DEDUPLICATE_TEXTS)
    X_train, y_train = get_X_y(splits["train"])
    X_valid, y_valid = get_X_y(splits["valid"])
    X_test, y_test = get_X_y(splits["test"])

    X_tr, X_va, X_te, pipe = _load_or_build_tfidf(X_train, X_valid, X_test)
    feature_names = pipe.vectorizer.get_feature_names_out()

    print(f"Train: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    print("Tuning C on validation set...")
    result = train_and_tune(X_tr, y_train, X_va, y_valid)
    model = result["model"]
    best_c = result["best_C"]
    thresholds = result["thresholds"]

    SVM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result["c_search"].to_csv(SVM_RESULTS_DIR / "c_search.csv", index=False)
    result["valid_metrics"]["per_label"].to_csv(
        SVM_RESULTS_DIR / "validation_per_label.csv", index=False
    )

    print(f"Best C = {best_c}")
    print(
        f"Validation macro-F1 = {result['valid_metrics']['macro_f1']:.4f}, "
        f"micro-F1 = {result['valid_metrics']['micro_f1']:.4f}"
    )
    print(f"Per-label thresholds: {dict(zip(LABEL_NAMES, thresholds.round(3)))}")

    baseline_valid = evaluate_baseline(X_tr, y_train, X_va, y_valid)
    baseline_test = evaluate_baseline(X_tr, y_train, X_te, y_test)

    y_test_pred = predict_multilabel(model, X_te, thresholds=thresholds)
    test_metrics = compute_multilabel_metrics(y_test, y_test_pred)

    save_svm_artifacts(
        model,
        thresholds,
        best_c,
        directory=SVM_MODEL_DIR,
        extra={"random_seed": RANDOM_SEED, "n_features": int(X_tr.shape[1])},
    )

    test_metrics["per_label"].to_csv(
        SVM_RESULTS_DIR / "test_per_label.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model": "svm_br",
                "split": "test",
                "best_C": best_c,
                "macro_f1": test_metrics["macro_f1"],
                "micro_f1": test_metrics["micro_f1"],
                "hamming_loss": test_metrics["hamming_loss"],
                "subset_accuracy": test_metrics["subset_accuracy"],
            },
            {
                "model": "baseline_majority",
                "split": "test",
                "best_C": None,
                "macro_f1": baseline_test["metrics"]["macro_f1"],
                "micro_f1": baseline_test["metrics"]["micro_f1"],
                "hamming_loss": baseline_test["metrics"]["hamming_loss"],
                "subset_accuracy": baseline_test["metrics"]["subset_accuracy"],
            },
        ]
    ).to_csv(SVM_RESULTS_DIR / "test_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "baseline_majority",
                "split": "valid",
                "macro_f1": baseline_valid["metrics"]["macro_f1"],
                "micro_f1": baseline_valid["metrics"]["micro_f1"],
            }
        ]
    ).to_csv(SVM_RESULTS_DIR / "baseline_valid.csv", index=False)

    top_tfidf_coefficients(model, feature_names).to_csv(
        SVM_RESULTS_DIR / "top_tfidf_coefficients.csv", index=False
    )

    report = {
        "method": "Binary Relevance + LinearSVC",
        "best_C": best_c,
        "thresholds": {name: float(t) for name, t in zip(LABEL_NAMES, thresholds)},
        "validation": _metrics_to_float(result["valid_metrics"]),
        "test": _metrics_to_float(test_metrics),
        "baseline_test": _metrics_to_float(baseline_test["metrics"]),
        "random_seed": RANDOM_SEED,
    }
    with (SVM_RESULTS_DIR / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- Test set (tuned thresholds) ---")
    print(f"  macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  micro-F1: {test_metrics['micro_f1']:.4f}")
    print(f"  hamming loss: {test_metrics['hamming_loss']:.4f}")
    print(f"  subset accuracy: {test_metrics['subset_accuracy']:.4f}")
    print("\n--- Baseline (majority per label) ---")
    print(f"  macro-F1: {baseline_test['metrics']['macro_f1']:.4f}")
    print(f"  micro-F1: {baseline_test['metrics']['micro_f1']:.4f}")
    print(f"\nSaved model to {SVM_MODEL_DIR}")
    print(f"Saved results to {SVM_RESULTS_DIR}")


if __name__ == "__main__":
    main()
