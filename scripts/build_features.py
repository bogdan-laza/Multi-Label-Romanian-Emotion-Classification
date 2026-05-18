#Laza Bogdan
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEDUPLICATE_TEXTS, MODELS_DIR, RANDOM_SEED  # noqa: E402
from src.data import get_X_y, load_all_splits  # noqa: E402
from src.features import TfidfFeaturePipeline  # noqa: E402


def main() -> None:
    splits = load_all_splits(deduplicate=DEDUPLICATE_TEXTS)
    X_train, y_train = get_X_y(splits["train"])
    X_valid, _ = get_X_y(splits["valid"])
    X_test, _ = get_X_y(splits["test"])

    use_svd = "--svd" in sys.argv
    pipe = TfidfFeaturePipeline(use_svd=use_svd)
    X_tr = pipe.fit_transform_train(X_train)
    X_va = pipe.transform(X_valid)
    X_te = pipe.transform(X_test)

    out_dir = pipe.save(MODELS_DIR / ("tfidf_svd" if use_svd else "tfidf"))
    print(f"Fitted TF-IDF{' + SVD' if use_svd else ''} on {len(X_train)} train tweets")
    print(f"  train matrix: {X_tr.shape}")
    print(f"  valid matrix: {X_va.shape}")
    print(f"  test matrix:  {X_te.shape}")
    print(f"  vocabulary size: {len(pipe.vectorizer.vocabulary_):,}")
    print(f"Saved pipeline to {out_dir}")
    print(f"Random seed (project): {RANDOM_SEED}")


if __name__ == "__main__":
    main()
