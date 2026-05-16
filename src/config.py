"""Project-wide constants, paths, and TF-IDF defaults."""

from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Paths (project root = parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "redv2"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Official REDv2 label order in JSON arrays (see REDv2 README)
REDV2_JSON_LABEL_ORDER = [
    "Sadness",
    "Surprise",
    "Fear",
    "Anger",
    "Neutral",
    "Trust",
    "Joy",
]

# Canonical order used in this project (README Section 2.1)
LABEL_NAMES = [
    "Anger",
    "Fear",
    "Joy",
    "Sadness",
    "Neutral",
    "Trust",
    "Surprise",
]

# Map REDv2 JSON index -> project LABEL_NAMES index
JSON_TO_PROJECT_LABEL_INDEX = [
    LABEL_NAMES.index(name) for name in REDV2_JSON_LABEL_ORDER
]

# Plutchik basic emotions (reference only; REDv2 has no Disgust / Anticipation)
PLUTCHIK_BASIC_EMOTIONS = [
    "Joy",
    "Trust",
    "Fear",
    "Surprise",
    "Sadness",
    "Disgust",
    "Anger",
    "Anticipation",
]

PLUTCHIK_IN_REDV2 = {
    "Anger": True,
    "Fear": True,
    "Joy": True,
    "Sadness": True,
    "Trust": True,
    "Surprise": True,
    "Neutral": False,
    "Disgust": False,
    "Anticipation": False,
}

# REDv2 official splits (train / valid / test)
SPLITS = ("train", "valid", "test")
SPLIT_FILES = {
    "train": "train.json",
    "valid": "valid.json",
    "test": "test.json",
}

REDV2_RAW_URLS = {
    split: (
        "https://raw.githubusercontent.com/Alegzandra/"
        f"RED-Romanian-Emotion-Datasets/main/REDv2/data/{fname}"
    )
    for split, fname in SPLIT_FILES.items()
}

# Text preprocessing (documented in report)
LOWERCASE_TEXT = True
STRIP_WHITESPACE = True

# If True, drop duplicate raw tweets (train > valid > test priority)
DEDUPLICATE_TEXTS = True

# TF-IDF defaults (Section 2.4) — tune on validation later
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

# Optional SVD for MLP (fit on train only; Task 2)
SVD_N_COMPONENTS = 300

# Method 1: Binary Relevance + Linear SVM (Section 3.1)
SVM_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SVM_MAX_ITER = 5000
SVM_CLASS_WEIGHT = "balanced"
SVM_MODEL_DIR = MODELS_DIR / "svm_br"
SVM_RESULTS_DIR = RESULTS_DIR / "svm_br"
SVM_TOP_COEF_K = 20
