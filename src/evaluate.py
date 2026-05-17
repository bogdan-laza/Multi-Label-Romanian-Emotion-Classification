from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from src.config import LABEL_NAMES


def binarize_proba(y_proba: np.ndarray, threshold: float | np.ndarray = 0.5) -> np.ndarray:
    if np.isscalar(threshold):
        return (y_proba >= threshold).astype(np.int8)
    threshold = np.asarray(threshold).reshape(1, -1)
    return (y_proba >= threshold).astype(np.int8)


def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.all(y_true == y_pred, axis=1).mean())


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, float | pd.DataFrame]:
    label_names = label_names or LABEL_NAMES

    metrics: dict[str, float | pd.DataFrame] = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": subset_accuracy(y_true, y_pred),
    }

    per_label = []
    for i, name in enumerate(label_names):
        per_label.append(
            {
                "label": name,
                "precision": precision_score(
                    y_true[:, i], y_pred[:, i], zero_division=0
                ),
                "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                "support": int(y_true[:, i].sum()),
            }
        )
    metrics["per_label"] = pd.DataFrame(per_label)
    return metrics


def tune_per_label_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)

    n_labels = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5, dtype=np.float64)
    for j in range(n_labels):
        best_f1 = -1.0
        for t in grid:
            pred = (y_proba[:, j] >= t).astype(np.int8)
            f1 = f1_score(y_true[:, j], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                thresholds[j] = t
    return thresholds
