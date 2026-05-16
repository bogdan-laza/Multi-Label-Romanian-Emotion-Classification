"""Load REDv2 splits and expose texts + multi-label targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    JSON_TO_PROJECT_LABEL_INDEX,
    LABEL_NAMES,
    REDV2_JSON_LABEL_ORDER,
    SPLIT_FILES,
    SPLITS,
)
from src.preprocess import preprocess_texts


def _reorder_labels(label_vector: list[int]) -> list[int]:
    """Map REDv2 JSON label order to project LABEL_NAMES order."""
    arr = np.asarray(label_vector, dtype=np.int8)
    return arr[JSON_TO_PROJECT_LABEL_INDEX].tolist()


def load_raw_split(split: str, data_dir: Path | None = None) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")

    data_dir = data_dir or DATA_DIR
    path = data_dir / SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/download_data.py"
        )

    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array")
    return records


def records_to_dataframe(records: list[dict], preprocess: bool = True) -> pd.DataFrame:
    rows = []
    for rec in records:
        if "agreed_labels" not in rec or "text" not in rec:
            raise KeyError("Each record must have 'text' and 'agreed_labels'")

        labels = _reorder_labels(rec["agreed_labels"])
        row = {
            "text_id": rec.get("text_id"),
            "text_raw": rec["text"],
            "text": rec["text"],
        }
        for name, val in zip(LABEL_NAMES, labels):
            row[name] = int(val)
        rows.append(row)

    df = pd.DataFrame(rows)
    if preprocess:
        df["text"] = preprocess_texts(df["text"].tolist())
    return df


def load_split(split: str, data_dir: Path | None = None, preprocess: bool = True) -> pd.DataFrame:
    return records_to_dataframe(load_raw_split(split, data_dir), preprocess=preprocess)


def deduplicate_splits(
    splits: dict[str, pd.DataFrame],
    split_priority: tuple[str, ...] = ("train", "valid", "test"),
) -> dict[str, pd.DataFrame]:
    """
    Remove duplicate raw tweet text.

    - Within a split: keep the first occurrence.
    - Across splits: keep the row in the highest-priority split only
      (default: train > valid > test).
    """
    seen: set[str] = set()
    priority_rank = {name: i for i, name in enumerate(split_priority)}
    out: dict[str, pd.DataFrame] = {}

    for split_name in sorted(splits, key=lambda s: priority_rank.get(s, 99)):
        df = splits[split_name].copy()
        mask_keep = []
        for text in df["text_raw"].astype(str):
            if text in seen:
                mask_keep.append(False)
            else:
                seen.add(text)
                mask_keep.append(True)
        out[split_name] = df.loc[mask_keep].reset_index(drop=True)

    return {name: out[name] for name in SPLITS if name in out}


def load_all_splits(
    data_dir: Path | None = None,
    preprocess: bool = True,
    deduplicate: bool = False,
) -> dict[str, pd.DataFrame]:
    splits = {split: load_split(split, data_dir, preprocess) for split in SPLITS}
    if deduplicate:
        splits = deduplicate_splits(splits)
    return splits


def get_X_y(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    X = df["text"].tolist()
    y = df[LABEL_NAMES].values.astype(np.int8)
    return X, y


def check_duplicates_across_splits(
    splits: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """
    Detect duplicate tweet text across official splits.

    Returns summary dict with duplicate text keys and counts per split pair.
    """
    text_to_splits: dict[str, set[str]] = {}
    for split_name, df in splits.items():
        for text in df["text_raw"].astype(str):
            text_to_splits.setdefault(text, set()).add(split_name)

    cross_split_dupes = {
        text: sorted(split_set)
        for text, split_set in text_to_splits.items()
        if len(split_set) > 1
    }

    within_split_dupes: dict[str, int] = {}
    for split_name, df in splits.items():
        n_dup = int(df["text_raw"].duplicated().sum())
        if n_dup:
            within_split_dupes[split_name] = n_dup

    return {
        "cross_split_duplicate_texts": len(cross_split_dupes),
        "cross_split_examples": dict(list(cross_split_dupes.items())[:5]),
        "within_split_duplicates": within_split_dupes,
    }


def dataset_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        labels_per_tweet = df[LABEL_NAMES].sum(axis=1)
        rows.append(
            {
                "split": name,
                "n_samples": len(df),
                "n_unique_text_id": df["text_id"].nunique(),
                "avg_labels_per_tweet": labels_per_tweet.mean(),
                "pct_multi_label": (labels_per_tweet > 1).mean() * 100,
                "avg_char_length": df["text_raw"].str.len().mean(),
            }
        )
    return pd.DataFrame(rows)


def verify_labels(splits: dict[str, pd.DataFrame]) -> None:
    """Assert binary agreed_labels and expected columns."""
    for name, df in splits.items():
        for col in LABEL_NAMES:
            if col not in df.columns:
                raise ValueError(f"{name}: missing label column {col}")
            unique = set(df[col].unique())
            if not unique.issubset({0, 1}):
                raise ValueError(f"{name}.{col}: expected binary labels, got {unique}")


def label_cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Co-occurrence counts: how often label pairs appear on the same tweet."""
    y = df[LABEL_NAMES].values
    co = y.T @ y
    return pd.DataFrame(co, index=LABEL_NAMES, columns=LABEL_NAMES)


def export_label_mapping_table(path: Path) -> None:
    """Write REDv2 vs Plutchik mapping for the report (Section 2.2)."""
    from src.config import PLUTCHIK_IN_REDV2, PROJECT_ROOT

    rows = []
    for label in LABEL_NAMES:
        rows.append(
            {
                "REDv2_label": label,
                "In_Plutchik_8_basic": PLUTCHIK_IN_REDV2.get(label, False),
                "REDv2_JSON_index": REDV2_JSON_LABEL_ORDER.index(label)
                if label in REDV2_JSON_LABEL_ORDER
                else None,
            }
        )
    out = path or (PROJECT_ROOT / "results" / "label_mapping.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
