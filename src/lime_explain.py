"""LIME explanations for SVM vs MLP on the same test tweets (README Section 5)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.neural_network import MLPClassifier

from src.config import (
    LABEL_NAMES,
    LIME_NUM_FEATURES,
    LIME_NUM_SAMPLES,
    LIME_RESULTS_DIR,
    MLP_FEATURES_DIR,
    MLP_MODEL_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    SVM_MODEL_DIR,
)
from src.evaluate import binarize_proba
from src.features import TfidfFeaturePipeline
from src.train_svm import decision_scores, load_svm_artifacts, predict_multilabel


def _scores_to_binary_proba(scores: np.ndarray) -> np.ndarray:
    """Map decision scores to 2-column [absent, present] probabilities."""
    p = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    return np.column_stack([1.0 - p, p])


def svm_label_predict_proba(
    texts: list[str],
    label_idx: int,
    model,
    pipe: TfidfFeaturePipeline,
) -> np.ndarray:
    X = pipe.transform(texts)
    scores = decision_scores(model, X)[:, label_idx]
    return _scores_to_binary_proba(scores)


def mlp_label_predict_proba(
    texts: list[str],
    label_idx: int,
    mlp: MLPClassifier,
    pipe: TfidfFeaturePipeline,
) -> np.ndarray:
    X = pipe.transform(texts, apply_svd=True)
    proba = mlp.predict_proba(X)
    p = np.clip(proba[:, label_idx], 1e-7, 1.0 - 1e-7)
    return np.column_stack([1.0 - p, p])


def explain_label(
    text: str,
    predict_proba_fn,
    label_name: str,
    num_features: int = LIME_NUM_FEATURES,
    num_samples: int = LIME_NUM_SAMPLES,
):
    explainer = LimeTextExplainer(class_names=["Absent", label_name])
    return explainer.explain_instance(
        text,
        predict_proba_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=[1],
    )


def explanation_to_rows(
    case_id: int,
    text: str,
    model_name: str,
    label_name: str,
    explanation,
) -> list[dict]:
    rows = []
    for feature, weight in explanation.as_list(label=1):
        rows.append(
            {
                "case_id": case_id,
                "model": model_name,
                "label": label_name,
                "feature": feature,
                "weight": weight,
                "text_preview": text[:120],
            }
        )
    return rows


def select_case_studies(
    texts: list[str],
    y_true: np.ndarray,
    y_svm: np.ndarray,
    y_mlp: np.ndarray,
    svm_scores: np.ndarray,
    mlp_proba: np.ndarray,
    svm_thresholds: np.ndarray,
    mlp_thresholds: np.ndarray,
    n_cases: int = 8,
) -> pd.DataFrame:
    """
    Pick diverse test tweets: multi-label, disagreement, neutral+emotion, confidence.
    """
    n = len(texts)
    neutral_idx = LABEL_NAMES.index("Neutral")
    labels_true = [set(LABEL_NAMES[i] for i in range(len(LABEL_NAMES)) if y_true[j, i]) for j in range(n)]

    disagree = np.any(y_svm != y_mlp, axis=1)
    multi_label = y_true.sum(axis=1) >= 2
    neutral_plus = (y_true[:, neutral_idx] == 1) & (y_true.sum(axis=1) >= 2)

    svm_conf = np.abs(svm_scores - svm_thresholds.reshape(1, -1)).min(axis=1)
    mlp_conf = np.abs(mlp_proba - mlp_thresholds.reshape(1, -1)).min(axis=1)
    uncertain = (svm_conf < 0.3) | (mlp_conf < 0.15)
    confident = (svm_conf > 1.0) & (mlp_conf > 0.25)

    buckets: list[tuple[str, np.ndarray]] = [
        ("disagreement", disagree),
        ("multi_label", multi_label),
        ("neutral_plus_emotion", neutral_plus),
        ("uncertain", uncertain),
        ("confident", confident),
    ]

    chosen: list[int] = []
    reasons: dict[int, list[str]] = {}

    def add(idx: int, reason: str) -> None:
        if idx in chosen or len(chosen) >= n_cases:
            return
        chosen.append(idx)
        reasons.setdefault(idx, []).append(reason)

    for reason, mask in buckets:
        candidates = np.where(mask)[0]
        for idx in candidates:
            add(int(idx), reason)
            if len(chosen) >= n_cases:
                break

    if len(chosen) < n_cases:
        for idx in range(n):
            add(idx, "fill")
            if len(chosen) >= n_cases:
                break

    rows = []
    for case_id, idx in enumerate(chosen[:n_cases]):
        rows.append(
            {
                "case_id": case_id,
                "test_index": idx,
                "text": texts[idx],
                "true_labels": ", ".join(sorted(labels_true[idx])) or "(none)",
                "svm_pred": ", ".join(LABEL_NAMES[i] for i in range(len(LABEL_NAMES)) if y_svm[idx, i]),
                "mlp_pred": ", ".join(LABEL_NAMES[i] for i in range(len(LABEL_NAMES)) if y_mlp[idx, i]),
                "disagree": bool(disagree[idx]),
                "selection_reasons": "; ".join(reasons.get(idx, [])),
            }
        )
    return pd.DataFrame(rows)


def load_models() -> tuple:
    svm_model, svm_thresholds, svm_meta = load_svm_artifacts(SVM_MODEL_DIR)
    svm_pipe = TfidfFeaturePipeline.load(MODELS_DIR / "tfidf")

    mlp = joblib.load(MLP_MODEL_DIR / "mlp_classifier.joblib")
    mlp_thresholds = np.load(MLP_MODEL_DIR / "thresholds.npy")
    mlp_pipe = TfidfFeaturePipeline.load(MLP_FEATURES_DIR)

    return svm_model, svm_thresholds, svm_pipe, mlp, mlp_thresholds, mlp_pipe


def run_lime_comparison(
    texts: list[str],
    y_true: np.ndarray,
    cases_df: pd.DataFrame,
    svm_model,
    svm_thresholds: np.ndarray,
    svm_pipe: TfidfFeaturePipeline,
    mlp: MLPClassifier,
    mlp_thresholds: np.ndarray,
    mlp_pipe: TfidfFeaturePipeline,
    output_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or LIME_RESULTS_DIR
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for _, case in cases_df.iterrows():
        case_id = int(case["case_id"])
        text = str(case["text"])
        idx = int(case["test_index"])
        def _pred_labels(field: str) -> set[str]:
            raw = str(case.get(field, "")).strip()
            if not raw:
                return set()
            return {ln for ln in LABEL_NAMES if ln in raw.split(", ")}

        case_labels = (
            {ln for ln in LABEL_NAMES if y_true[idx, LABEL_NAMES.index(ln)]}
            | _pred_labels("svm_pred")
            | _pred_labels("mlp_pred")
        )

        for label_name in sorted(case_labels):
            label_idx = LABEL_NAMES.index(label_name)

            svm_fn = lambda t, li=label_idx: svm_label_predict_proba(
                t, li, svm_model, svm_pipe
            )
            mlp_fn = lambda t, li=label_idx: mlp_label_predict_proba(
                t, li, mlp, mlp_pipe
            )

            svm_exp = explain_label(text, svm_fn, label_name)
            mlp_exp = explain_label(text, mlp_fn, label_name)

            all_rows.extend(
                explanation_to_rows(case_id, text, "svm", label_name, svm_exp)
            )
            all_rows.extend(
                explanation_to_rows(case_id, text, "mlp", label_name, mlp_exp)
            )

            safe_label = label_name.lower()
            (html_dir / f"case{case_id}_{safe_label}_svm.html").write_text(
                svm_exp.as_html(), encoding="utf-8"
            )
            (html_dir / f"case{case_id}_{safe_label}_mlp.html").write_text(
                mlp_exp.as_html(), encoding="utf-8"
            )

    features_df = pd.DataFrame(all_rows)
    features_df.to_csv(output_dir / "lime_feature_weights.csv", index=False)
    cases_df.to_csv(output_dir / "case_studies.csv", index=False)

    meta = {
        "random_seed": RANDOM_SEED,
        "num_cases": len(cases_df),
        "lime_num_features": LIME_NUM_FEATURES,
        "lime_num_samples": LIME_NUM_SAMPLES,
    }
    with (output_dir / "lime_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return output_dir
