"""TF-IDF vectorization (fit on train only) and optional TruncatedSVD for MLP."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    MODELS_DIR,
    RANDOM_SEED,
    SVD_N_COMPONENTS,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)


def build_tfidf_vectorizer(
    max_features: int | None = None,
    ngram_range: tuple[int, int] | None = None,
    min_df: int | float | None = None,
    max_df: float | None = None,
    sublinear_tf: bool | None = None,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features if max_features is not None else TFIDF_MAX_FEATURES,
        ngram_range=ngram_range if ngram_range is not None else TFIDF_NGRAM_RANGE,
        min_df=min_df if min_df is not None else TFIDF_MIN_DF,
        max_df=max_df if max_df is not None else TFIDF_MAX_DF,
        sublinear_tf=sublinear_tf if sublinear_tf is not None else TFIDF_SUBLINEAR_TF,
        dtype=np.float32,
    )


class TfidfFeaturePipeline:
    """
    Fit TF-IDF on training texts only; transform train/valid/test.

    Optional SVD branch for MLP (fit SVD on train TF-IDF only).
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer | None = None,
        svd: TruncatedSVD | None = None,
        use_svd: bool = False,
        svd_components: int | None = None,
    ):
        self.vectorizer = vectorizer or build_tfidf_vectorizer()
        self.use_svd = use_svd
        self.svd_components = svd_components or SVD_N_COMPONENTS
        self.svd = svd
        self._fitted = False

    def fit(self, X_train: list[str]) -> "TfidfFeaturePipeline":
        self.vectorizer.fit(X_train)
        if self.use_svd:
            self.svd = TruncatedSVD(
                n_components=self.svd_components,
                random_state=RANDOM_SEED,
            )
            X_train_tfidf = self.vectorizer.transform(X_train)
            self.svd.fit(X_train_tfidf)
        self._fitted = True
        return self

    def transform(self, X: list[str], apply_svd: bool | None = None) -> csr_matrix | np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() on training data first")
        X_tfidf = self.vectorizer.transform(X)
        use_svd = self.use_svd if apply_svd is None else apply_svd
        if use_svd:
            if self.svd is None:
                raise RuntimeError("SVD requested but not fitted")
            return self.svd.transform(X_tfidf)
        return X_tfidf

    def fit_transform_train(
        self, X_train: list[str], apply_svd: bool | None = None
    ) -> csr_matrix | np.ndarray:
        self.fit(X_train)
        return self.transform(X_train, apply_svd=apply_svd)

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or (MODELS_DIR / "tfidf")
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "vectorizer.joblib")
        meta = {"use_svd": self.use_svd, "svd_components": self.svd_components}
        joblib.dump(meta, directory / "meta.joblib")
        if self.svd is not None:
            joblib.dump(self.svd, directory / "svd.joblib")
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "TfidfFeaturePipeline":
        directory = directory or (MODELS_DIR / "tfidf")
        vectorizer = joblib.load(directory / "vectorizer.joblib")
        meta = joblib.load(directory / "meta.joblib")
        svd_path = directory / "svd.joblib"
        svd = joblib.load(svd_path) if svd_path.is_file() else None
        pipe = cls(
            vectorizer=vectorizer,
            svd=svd,
            use_svd=meta["use_svd"],
            svd_components=meta["svd_components"],
        )
        pipe._fitted = True
        return pipe


def fit_tfidf_splits(
    X_train: list[str],
    X_valid: list[str],
    X_test: list[str],
    use_svd: bool = False,
) -> tuple[csr_matrix | np.ndarray, ...]:
    """Convenience: fit on train, return (X_train, X_valid, X_test) features."""
    pipe = TfidfFeaturePipeline(use_svd=use_svd)
    X_tr = pipe.fit_transform_train(X_train, apply_svd=use_svd)
    X_va = pipe.transform(X_valid, apply_svd=use_svd)
    X_te = pipe.transform(X_test, apply_svd=use_svd)
    return X_tr, X_va, X_te, pipe
