#Laza Bogdan
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, LABEL_NAMES, REDV2_JSON_LABEL_ORDER, RESULTS_DIR  # noqa: E402
from src.data import (  # noqa: E402
    check_duplicates_across_splits,
    dataset_summary,
    export_label_mapping_table,
    load_all_splits,
    verify_labels,
)


def main() -> None:
    splits = load_all_splits()
    verify_labels(splits)

    print("=== REDv2 label order (JSON) ===")
    print(REDV2_JSON_LABEL_ORDER)
    print("\n=== Project label columns ===")
    print(LABEL_NAMES)

    summary = dataset_summary(splits)
    print("\n=== Split summary ===")
    print(summary.to_string(index=False))

    dupes = check_duplicates_across_splits(splits)
    print("\n=== Duplicate check ===")
    print(json.dumps(dupes, indent=2, ensure_ascii=True))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "split_summary.csv"
    summary.to_csv(summary_path, index=False)
    export_label_mapping_table(RESULTS_DIR / "label_mapping.csv")

    print(f"\nWrote {summary_path}")
    print(f"Wrote {RESULTS_DIR / 'label_mapping.csv'}")
    if dupes["cross_split_duplicate_texts"] or dupes["within_split_duplicates"]:
        print(
            "\nNote: duplicates found. For modeling, use "
            "load_all_splits(deduplicate=True) or set DEDUPLICATE_TEXTS in config."
        )

    print("\nVerification OK.")


if __name__ == "__main__":
    main()
