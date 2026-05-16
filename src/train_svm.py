"""Method 1: Binary Relevance with linear SVM on TF-IDF features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from src.config import (
    LABEL_NAMES,
    RANDOM_SEED,
    SVM_C_GRID,
    SVM_CLASS_WEIGHT,
    SVM_MAX_ITER,
    SVM_MODEL_DIR,
    SVM_TOP_COEF_K,
)
from src.evaluate import (
    binarize_proba,
    compute_multilabel_metrics,
    tune_per_label_thresholds,
)


def build_linear_svm_br(
    C: float = 1.0,
    class_weight: str | dict | None = None,
    max_iter: int | None = None,
    random_state: int | None = None,
) -> OneVsRestClassifier:
    """One binary LinearSVC per label (Binary Relevance)."""
    base = LinearSVC(
        C=C,
        class_weight=class_weight if class_weight is not None else SVM_CLASS_WEIGHT,
        max_iter=max_iter if max_iter is not None else SVM_MAX_ITER,
        random_state=random_state if random_state is not None else RANDOM_SEED,
        dual="auto",
    )
    return OneVsRestClassifier(base, n_jobs=-1)


def build_majority_baseline() -> OneVsRestClassifier:
    """Per-label majority class (README Section 3.3)."""
    return OneVsRestClassifier(
        DummyClassifier(strategy="most_frequent"),
        n_jobs=-1,
    )


def decision_scores(model: OneVsRestClassifier, X) -> np.ndarray:
    """Per-label decision scores (higher => more likely positive)."""
    return model.decision_function(X)


def tune_svm_c(
    X_train,
    y_train: np.ndarray,
    X_valid,
    y_valid: np.ndarray,
    c_grid: list[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Pick regularization C by validation macro-F1 (default threshold 0 on scores).
    """
    c_grid = c_grid if c_grid is not None else list(SVM_C_GRID)
    rows = []
    best_c = c_grid[0]
    best_macro = -1.0

    for C in c_grid:
        model = build_linear_svm_br(C=C)
        model.fit(X_train, y_train)
        scores = decision_scores(model, X_valid)
        y_pred = (scores >= 0.0).astype(np.int8)
        metrics = compute_multilabel_metrics(y_valid, y_pred)
        macro_f1 = float(metrics["macro_f1"])
        rows.append(
            {
                "C": C,
                "macro_f1": macro_f1,
                "micro_f1": float(metrics["micro_f1"]),
                "hamming_loss": float(metrics["hamming_loss"]),
            }
        )
        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_c = C

    return best_c, pd.DataFrame(rows)


def train_svm_br(
    X_train,
    y_train: np.ndarray,
    C: float = 1.0,
) -> OneVsRestClassifier:
    model = build_linear_svm_br(C=C)
    model.fit(X_train, y_train)
    return model


def train_and_tune(
    X_train,
    y_train: np.ndarray,
    X_valid,
    y_valid: np.ndarray,
    c_grid: list[float] | None = None,
    threshold_grid: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Full Method 1 pipeline: C tuning, fit on train, per-label thresholds on valid.
    """
    best_c, c_search_df = tune_svm_c(
        X_train, y_train, X_valid, y_valid, c_grid=c_grid
    )
    model = train_svm_br(X_train, y_train, C=best_c)

    valid_scores = decision_scores(model, X_valid)
    thresholds = tune_per_label_thresholds(
        y_valid, valid_scores, grid=threshold_grid
    )
    valid_pred = binarize_proba(valid_scores, threshold=thresholds)
    valid_metrics = compute_multilabel_metrics(y_valid, valid_pred)

    return {
        "model": model,
        "best_C": best_c,
        "c_search": c_search_df,
        "thresholds": thresholds,
        "valid_scores": valid_scores,
        "valid_metrics": valid_metrics,
    }


def evaluate_baseline(
    X_train,
    y_train: np.ndarray,
    X_split,
    y_split: np.ndarray,
) -> dict[str, object]:
    baseline = build_majority_baseline()
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_split).astype(np.int8)
    return {
        "model": baseline,
        "metrics": compute_multilabel_metrics(y_split, y_pred),
        "y_pred": y_pred,
    }


def top_tfidf_coefficients(
    model: OneVsRestClassifier,
    feature_names: np.ndarray,
    label_names: list[str] | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Top positive/negative TF-IDF weights per label (global linear interpretability).
    """
    label_names = label_names or LABEL_NAMES
    top_k = top_k if top_k is not None else SVM_TOP_COEF_K
    rows = []

    for label_idx, label in enumerate(label_names):
        estimator = model.estimators_[label_idx]
        coef = np.asarray(estimator.coef_).ravel()
        order_pos = np.argsort(coef)[::-1][:top_k]
        order_neg = np.argsort(coef)[:top_k]
        for rank, idx in enumerate(order_pos, start=1):
            rows.append(
                {
                    "label": label,
                    "sign": "positive",
                    "rank": rank,
                    "feature": feature_names[idx],
                    "coefficient": float(coef[idx]),
                }
            )
        for rank, idx in enumerate(order_neg, start=1):
            rows.append(
                {
                    "label": label,
                    "sign": "negative",
                    "rank": rank,
                    "feature": feature_names[idx],
                    "coefficient": float(coef[idx]),
                }
            )

    return pd.DataFrame(rows)


def save_svm_artifacts(
    model: OneVsRestClassifier,
    thresholds: np.ndarray,
    best_c: float,
    directory: Path | None = None,
    extra: dict | None = None,
) -> Path:
    directory = directory or SVM_MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "svm_br.joblib")
    joblib.dump(thresholds, directory / "thresholds.joblib")
    meta = {"best_C": best_c, "labels": LABEL_NAMES}
    if extra:
        meta.update(extra)
    joblib.dump(meta, directory / "meta.joblib")
    return directory


def load_svm_artifacts(directory: Path | None = None) -> tuple[OneVsRestClassifier, np.ndarray, dict]:
    directory = directory or SVM_MODEL_DIR
    model = joblib.load(directory / "svm_br.joblib")
    thresholds = joblib.load(directory / "thresholds.joblib")
    meta = joblib.load(directory / "meta.joblib")
    return model, thresholds, meta


def predict_multilabel(
    model: OneVsRestClassifier,
    X,
    thresholds: np.ndarray | float = 0.0,
) -> np.ndarray:
    scores = decision_scores(model, X)
    return binarize_proba(scores, threshold=thresholds)
