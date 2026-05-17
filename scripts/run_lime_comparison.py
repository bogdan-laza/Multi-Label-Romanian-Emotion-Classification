from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEDUPLICATE_TEXTS,
    LIME_NUM_CASES,
    LIME_RESULTS_DIR,
    MLP_FEATURES_DIR,
    MLP_MODEL_DIR,
    MODELS_DIR,
    SVM_MODEL_DIR,
)
from src.data import get_X_y, load_all_splits  # noqa: E402
from src.evaluate import binarize_proba  # noqa: E402
from src.features import TfidfFeaturePipeline  # noqa: E402
from src.lime_explain import (  # noqa: E402
    load_models,
    run_lime_comparison,
    select_case_studies,
)
from src.train_svm import decision_scores, predict_multilabel  # noqa: E402


def main() -> None:
    for path, name in [
        (SVM_MODEL_DIR / "svm_br.joblib", "SVM"),
        (MLP_MODEL_DIR / "mlp_classifier.joblib", "MLP"),
        (MODELS_DIR / "tfidf" / "vectorizer.joblib", "TF-IDF"),
        (MLP_FEATURES_DIR / "vectorizer.joblib", "TF-IDF+SVD"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} artifact at {path}. "
                "Run: train_svm.py, build_features.py --svd, train_mlp.py"
            )

    splits = load_all_splits(deduplicate=DEDUPLICATE_TEXTS)
    X_test, y_test = get_X_y(splits["test"])

    svm_model, svm_thresholds, svm_pipe, mlp, mlp_thresholds, mlp_pipe = load_models()

    X_svm = svm_pipe.transform(X_test)
    svm_scores = decision_scores(svm_model, X_svm)
    y_svm = predict_multilabel(svm_model, X_svm, thresholds=svm_thresholds)

    X_mlp = mlp_pipe.transform(X_test, apply_svd=True)
    mlp_proba = mlp.predict_proba(X_mlp)
    y_mlp = binarize_proba(mlp_proba, mlp_thresholds)

    cases = select_case_studies(
        X_test,
        y_test,
        y_svm,
        y_mlp,
        svm_scores,
        mlp_proba,
        svm_thresholds,
        mlp_thresholds,
        n_cases=LIME_NUM_CASES,
    )

    print(f"Selected {len(cases)} case studies for LIME:")
    for _, row in cases.iterrows():
        print(
            f"  case {row['case_id']}: disagree={row['disagree']} "
            f"| true=[{row['true_labels']}]"
        )

    out = run_lime_comparison(
        X_test,
        y_test,
        cases,
        svm_model,
        svm_thresholds,
        svm_pipe,
        mlp,
        mlp_thresholds,
        mlp_pipe,
    )
    print(f"\nLIME outputs written to {out}")
    print(f"  case_studies.csv, lime_feature_weights.csv, html/")


if __name__ == "__main__":
    main()
