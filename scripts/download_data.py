from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, REDV2_RAW_URLS, SPLIT_FILES  # noqa: E402


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")


def main() -> None:
    for split, url in REDV2_RAW_URLS.items():
        dest = DATA_DIR / SPLIT_FILES[split]
        if dest.is_file():
            print(f"Skip {dest.name} (already exists)")
            continue
        download_file(url, dest)
    print("Done. REDv2 data is in:", DATA_DIR)


if __name__ == "__main__":
    main()
