"""
Train multi-label MLP on TF-IDF + TruncatedSVD features (REDv2).

Uses the shared TfidfFeaturePipeline (use_svd=True, 300 dims), sklearn MLPClassifier
with sigmoid outputs (binary cross-entropy), early stopping on the official validation
split, and per-label threshold tuning via src.evaluate.
"""

from __future__ import annotations

import json
import sys
import warnings
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEDUPLICATE_TEXTS,
    LABEL_NAMES,
    MLP_RESULTS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    SVD_N_COMPONENTS,
)
from src.data import get_X_y, load_all_splits  # noqa: E402
from src.evaluate import (  # noqa: E402
    binarize_proba,
    compute_multilabel_metrics,
    tune_per_label_thresholds,
)
from src.features import TfidfFeaturePipeline  # noqa: E402

# Feature pipeline (must match build_features.py --svd)
FEATURES_DIR = MODELS_DIR / "tfidf_svd"
MLP_DIR = MODELS_DIR / "mlp"

# MLP hyperparameters (README Section 3.2)
HIDDEN_LAYER_SIZES = (256, 128)
MLP_ALPHA = 1e-4
LEARNING_RATE_INIT = 1e-3
BATCH_SIZE = 64
ITERATIONS_PER_EPOCH = 20
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4


def multilabel_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean binary cross-entropy across all labels and samples."""
    if not np.isfinite(y_proba).all():
        return float("inf")
    eps = 1e-7
    p = np.clip(y_proba, eps, 1.0 - eps)
    loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(np.nanmean(loss))


def load_or_fit_features(
    X_train: list[str],
    X_valid: list[str],
    X_test: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TfidfFeaturePipeline]:
    """Load pre-fitted TF-IDF + SVD pipeline, or fit and save if missing."""
    if FEATURES_DIR.is_dir() and (FEATURES_DIR / "vectorizer.joblib").is_file():
        pipe = TfidfFeaturePipeline.load(FEATURES_DIR)
        if not pipe.use_svd:
            raise ValueError(
                f"{FEATURES_DIR} was saved without SVD. "
                "Run: python scripts/build_features.py --svd"
            )
        print(f"Loaded feature pipeline from {FEATURES_DIR}")
    else:
        print(f"No pipeline at {FEATURES_DIR}; fitting TF-IDF + SVD on train ...")
        pipe = TfidfFeaturePipeline(use_svd=True, svd_components=SVD_N_COMPONENTS)
        pipe.fit(X_train)
        pipe.save(FEATURES_DIR)

    X_tr = pipe.transform(X_train, apply_svd=True)
    X_va = pipe.transform(X_valid, apply_svd=True)
    X_te = pipe.transform(X_test, apply_svd=True)
    return X_tr, X_va, X_te, pipe


def build_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=MLP_ALPHA,
        batch_size=BATCH_SIZE,
        learning_rate="adaptive",
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=ITERATIONS_PER_EPOCH,
        warm_start=True,
        random_state=RANDOM_SEED,
        early_stopping=False,
        verbose=False,
    )


def train_with_validation_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[MLPClassifier, dict]:
    """Train MLP with warm_start; monitor validation BCE and restore best weights."""
    mlp = build_mlp()
    best_val_loss = np.inf
    best_checkpoint: bytes | None = None
    patience = 0
    history: list[dict] = []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Stochastic Optimizer: Maximum iterations",
            category=UserWarning,
            module="sklearn.neural_network",
        )
        for epoch in range(1, MAX_EPOCHS + 1):
            mlp.fit(X_train, y_train)
            y_proba_valid = mlp.predict_proba(X_valid)
            val_loss = multilabel_log_loss(y_valid, y_proba_valid)
            history.append({"epoch": epoch, "val_log_loss": val_loss})

            if not np.isfinite(val_loss):
                print(f"  epoch {epoch:3d}  validation loss diverged; restoring best weights")
                break

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                buf = BytesIO()
                joblib.dump(mlp, buf)
                best_checkpoint = buf.getvalue()
                patience = 0
            else:
                patience += 1

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:3d}  val BCE={val_loss:.4f}  best={best_val_loss:.4f}"
                )

            if patience >= EARLY_STOPPING_PATIENCE:
                print(
                    f"  early stop at epoch {epoch} "
                    f"(patience={EARLY_STOPPING_PATIENCE})"
                )
                break

    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a checkpoint")

    mlp = joblib.load(BytesIO(best_checkpoint))
    return mlp, {
        "best_val_log_loss": best_val_loss,
        "epochs_run": len(history),
        "history": history,
    }


def metrics_to_serializable(metrics: dict) -> dict:
    out: dict = {}
    for key, val in metrics.items():
        if key == "per_label":
            out[key] = val.to_dict(orient="records")
        else:
            out[key] = float(val) if isinstance(val, (np.floating, float)) else val
    return out


def save_checkpoint(
    mlp: MLPClassifier,
    thresholds: np.ndarray,
    train_info: dict,
    feature_dir: Path,
) -> None:
    MLP_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp, MLP_DIR / "mlp_classifier.joblib")
    np.save(MLP_DIR / "thresholds.npy", thresholds)

    threshold_map = {
        label: float(thresholds[i]) for i, label in enumerate(LABEL_NAMES)
    }
    meta = {
        "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
        "solver": "adam",
        "alpha": MLP_ALPHA,
        "learning_rate_init": LEARNING_RATE_INIT,
        "batch_size": BATCH_SIZE,
        "svd_components": SVD_N_COMPONENTS,
        "feature_pipeline": str(feature_dir),
        "random_seed": RANDOM_SEED,
        "deduplicate_texts": DEDUPLICATE_TEXTS,
        "thresholds": threshold_map,
        "training": {
            "best_val_log_loss": train_info["best_val_log_loss"],
            "epochs_run": train_info["epochs_run"],
        },
    }
    with (MLP_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved MLP checkpoint to {MLP_DIR}")


def main() -> None:
    print("=== Multi-label MLP training (REDv2) ===")
    print(f"  deduplicate={DEDUPLICATE_TEXTS}  seed={RANDOM_SEED}")

    splits = load_all_splits(deduplicate=DEDUPLICATE_TEXTS)
    X_train, y_train = get_X_y(splits["train"])
    X_valid, y_valid = get_X_y(splits["valid"])
    X_test, y_test = get_X_y(splits["test"])

    X_tr, X_va, X_te, _pipe = load_or_fit_features(X_train, X_valid, X_test)
    print(f"  train features: {X_tr.shape}  valid: {X_va.shape}  test: {X_te.shape}")

    print("\nTraining MLP (sigmoid / BCE, early stopping on validation) ...")
    mlp, train_info = train_with_validation_early_stopping(X_tr, y_train, X_va, y_valid)

    y_proba_valid = mlp.predict_proba(X_va)
    thresholds = tune_per_label_thresholds(y_valid, y_proba_valid)
    print("\nPer-label thresholds (validation):")
    for label, t in zip(LABEL_NAMES, thresholds):
        print(f"  {label:10s} {t:.3f}")

    y_pred_valid = binarize_proba(y_proba_valid, thresholds)
    valid_metrics = compute_multilabel_metrics(y_valid, y_pred_valid)

    y_proba_test = mlp.predict_proba(X_te)
    y_pred_test = binarize_proba(y_proba_test, thresholds)
    test_metrics = compute_multilabel_metrics(y_test, y_pred_test)

    print("\n=== Validation metrics (tuned thresholds) ===")
    print(f"  micro-F1: {valid_metrics['micro_f1']:.4f}")
    print(f"  macro-F1: {valid_metrics['macro_f1']:.4f}")
    print(f"  hamming:  {valid_metrics['hamming_loss']:.4f}")
    print(f"  subset accuracy: {valid_metrics['subset_accuracy']:.4f}")

    print("\n=== Test metrics (tuned thresholds) ===")
    print(f"  micro-F1: {test_metrics['micro_f1']:.4f}")
    print(f"  macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  hamming:  {test_metrics['hamming_loss']:.4f}")
    print(f"  subset accuracy: {test_metrics['subset_accuracy']:.4f}")

    save_checkpoint(mlp, thresholds, train_info, FEATURES_DIR)

    MLP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "validation": metrics_to_serializable(valid_metrics),
        "test": metrics_to_serializable(test_metrics),
        "thresholds": {l: float(thresholds[i]) for i, l in enumerate(LABEL_NAMES)},
        "training": train_info,
    }
    # history can be large; keep last 5 epochs in summary only
    results["training"] = {
        "best_val_log_loss": train_info["best_val_log_loss"],
        "epochs_run": train_info["epochs_run"],
    }
    with (MLP_RESULTS_DIR / "mlp_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    test_metrics["per_label"].to_csv(MLP_RESULTS_DIR / "mlp_per_label_test.csv", index=False)
    print(f"\nWrote {MLP_RESULTS_DIR / 'mlp_metrics.json'}")
    print(f"Wrote {MLP_RESULTS_DIR / 'mlp_per_label_test.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
