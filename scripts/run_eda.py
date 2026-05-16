"""Exploratory data analysis: label frequencies, co-occurrence, tweet lengths."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABEL_NAMES, RANDOM_SEED, RESULTS_DIR  # noqa: E402
from src.data import label_cooccurrence_matrix, load_all_splits  # noqa: E402

EDA_DIR = RESULTS_DIR / "eda"


def plot_label_frequencies(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    train = splits["train"]
    freq = train[LABEL_NAMES].sum().sort_values(ascending=True)
    pct = (freq / len(train) * 100).round(2)

    fig, ax = plt.subplots(figsize=(8, 5))
    freq.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Number of tweets with label (train)")
    ax.set_title("REDv2 train — per-label frequency (agreed_labels)")
    for i, (label, count) in enumerate(freq.items()):
        ax.text(count + 5, i, f"{pct[label]:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "label_frequencies_train.png", dpi=150)
    plt.close(fig)

    freq.to_csv(out_dir / "label_frequencies_train.csv")


def plot_cooccurrence(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    co = label_cooccurrence_matrix(splits["train"])
    co_norm = co.astype(float)
    np.fill_diagonal(co_norm.values, 0)
    row_sums = co_norm.sum(axis=1).replace(0, np.nan)
    co_pct = co_norm.div(row_sums, axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(co, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Co-occurrence counts (train)")
    sns.heatmap(co_pct, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Co-occurrence % of row label (train)")
    fig.tight_layout()
    fig.savefig(out_dir / "label_cooccurrence_train.png", dpi=150)
    plt.close(fig)

    co.to_csv(out_dir / "label_cooccurrence_train.csv")


def plot_tweet_lengths(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, df in splits.items():
        lengths = df["text_raw"].str.len()
        ax.hist(lengths, bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Character length (raw tweet)")
    ax.set_ylabel("Density")
    ax.set_title("Tweet length distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tweet_length_distribution.png", dpi=150)
    plt.close(fig)

    rows = []
    for name, df in splits.items():
        lengths = df["text_raw"].str.len()
        rows.append(
            {
                "split": name,
                "mean": lengths.mean(),
                "median": lengths.median(),
                "p90": lengths.quantile(0.9),
                "max": lengths.max(),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "tweet_length_stats.csv", index=False)


def plot_labels_per_tweet(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    train = splits["train"]
    counts = train[LABEL_NAMES].sum(axis=1).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="coral")
    ax.set_xlabel("Number of active labels per tweet")
    ax.set_ylabel("Count")
    ax.set_title("Labels per tweet (train)")
    fig.tight_layout()
    fig.savefig(out_dir / "labels_per_tweet_train.png", dpi=150)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    np.random.seed(RANDOM_SEED)

    splits = load_all_splits()
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    plot_label_frequencies(splits, EDA_DIR)
    plot_cooccurrence(splits, EDA_DIR)
    plot_tweet_lengths(splits, EDA_DIR)
    plot_labels_per_tweet(splits, EDA_DIR)

    print(f"EDA figures saved to {EDA_DIR}")


if __name__ == "__main__":
    main()
