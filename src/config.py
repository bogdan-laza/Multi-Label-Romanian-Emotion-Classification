from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "redv2"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

REDV2_JSON_LABEL_ORDER = [
    "Sadness",
    "Surprise",
    "Fear",
    "Anger",
    "Neutral",
    "Trust",
    "Joy",
]

LABEL_NAMES = [
    "Anger",
    "Fear",
    "Joy",
    "Sadness",
    "Neutral",
    "Trust",
    "Surprise",
]

JSON_TO_PROJECT_LABEL_INDEX = [
    LABEL_NAMES.index(name) for name in REDV2_JSON_LABEL_ORDER
]

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

LOWERCASE_TEXT = True
STRIP_WHITESPACE = True

DEDUPLICATE_TEXTS = False

TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

SVD_N_COMPONENTS = 300

SVM_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SVM_MAX_ITER = 5000
SVM_CLASS_WEIGHT = "balanced"
SVM_MODEL_DIR = MODELS_DIR / "svm_br"
SVM_RESULTS_DIR = RESULTS_DIR / "svm_br"
SVM_TOP_COEF_K = 20

MLP_MODEL_DIR = MODELS_DIR / "mlp"
MLP_FEATURES_DIR = MODELS_DIR / "tfidf_svd"
MLP_RESULTS_DIR = RESULTS_DIR / "mlp"

LIME_RESULTS_DIR = RESULTS_DIR / "lime"
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 1000
LIME_NUM_CASES = 8
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparison"
