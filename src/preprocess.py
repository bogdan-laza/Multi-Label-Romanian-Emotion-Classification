import re

from src.config import LOWERCASE_TEXT, STRIP_WHITESPACE

PLACEHOLDER_PATTERN = re.compile(r"<\|[^|]+\|>")


def preprocess_text(text: str) -> str:
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
