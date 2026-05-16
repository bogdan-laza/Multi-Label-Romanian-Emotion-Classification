"""Text preprocessing for REDv2 tweets."""

import re

from src.config import LOWERCASE_TEXT, STRIP_WHITESPACE

# Placeholder tokens left by REDv2 anonymization (e.g. <|PERSON|>)
PLACEHOLDER_PATTERN = re.compile(r"<\|[^|]+\|>")


def preprocess_text(text: str) -> str:
    """
    Normalize a single tweet for TF-IDF.

    - Keeps Romanian diacritics (ă, â, î, ș, ț) as in REDv2.
    - Lowercases if configured.
    - Normalizes placeholder spacing; placeholders are kept as tokens.
    """
    if not isinstance(text, str):
        text = str(text)

    if STRIP_WHITESPACE:
        text = text.strip()

    text = PLACEHOLDER_PATTERN.sub(lambda m: m.group(0).lower(), text)

    if LOWERCASE_TEXT:
        text = text.lower()

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_texts(texts: list[str]) -> list[str]:
    return [preprocess_text(text) for text in texts]
