"""Text preprocessing utilities 
- lowercase normalization
- tokenization with NLTK word_tokenize when available
- configurable base + domain stopword removal
- spaCy-first lemmatization with NLTK fallback
- minimum token length and numeric token filtering
- corpus-level preprocessing with token threshold filtering
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional

import pandas as pd

from config.settings import DOMAIN_STOPWORDS, MIN_TOKEN_THRESHOLD

LOGGER = logging.getLogger(__name__)
_SPACY_NLP = None


def _get_nltk_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return set()


def _get_combined_stopwords(extra_stopwords: Optional[Iterable[str]] = None) -> set[str]:
    combined = _get_nltk_stopwords().union(DOMAIN_STOPWORDS)
    if extra_stopwords:
        combined.update({word.lower() for word in extra_stopwords})
    return combined


def _get_spacy_model():
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except Exception:
        _SPACY_NLP = False
    return _SPACY_NLP


def _nltk_lemmatize(tokens: List[str]) -> List[str]:
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    except Exception:
        return tokens


def clean_text(
    text: str,
    use_spacy: bool = True,
    stopwords: Optional[set[str]] = None,
    min_len: int = 2,
) -> List[str]:
    """Convert raw text into cleaned tokens."""
    if not text or not isinstance(text, str):
        return []

    normalized = re.sub(r"\s+", " ", text.lower().strip())
    stopwords = stopwords or _get_combined_stopwords()

    if use_spacy:
        nlp = _get_spacy_model()
        if nlp:
            tokens: list[str] = []
            for token in nlp(normalized):
                lemma = token.lemma_.lower().strip()
                if not lemma or not token.is_alpha:
                    continue
                if lemma in stopwords or len(lemma) < min_len:
                    continue
                tokens.append(lemma)
            return tokens
        LOGGER.warning("spaCy model unavailable; falling back to NLTK preprocessing")

    try:
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(normalized)
    except Exception:
        tokens = re.findall(r"[a-z]+", normalized)

    filtered = [
        token
        for token in tokens
        if token.isalpha() and token not in stopwords and len(token) >= min_len and not token.isdigit()
    ]
    return _nltk_lemmatize(filtered)


def preprocess_corpus(
    df: pd.DataFrame,
    text_column: str = "abstract",
    title_column: Optional[str] = "title",
    use_spacy: bool = True,
    min_token_threshold: int = MIN_TOKEN_THRESHOLD,
) -> pd.DataFrame:
    """Preprocess a DataFrame and mark rows that should be excluded from text modeling."""
    df = df.copy()
    if title_column and title_column in df.columns:
        combined = (df[title_column].fillna("") + " " + df[text_column].fillna("")).str.strip()
    else:
        combined = df[text_column].fillna("")

    token_lists = combined.apply(lambda value: clean_text(value, use_spacy=use_spacy))
    df["tokens"] = token_lists
    df["cleaned_text"] = token_lists.apply(lambda tokens: " ".join(tokens))
    df["token_count"] = token_lists.apply(len)

    if "has_abstract" not in df.columns:
        df["has_abstract"] = df[text_column].fillna("").astype(str).str.len().ge(50)

    df["eligible_for_text_analysis"] = df["has_abstract"] & df["token_count"].ge(min_token_threshold)
    excluded = int((~df["eligible_for_text_analysis"]).sum())
    LOGGER.info("Excluded %s papers from text-based steps", excluded)
    return df
