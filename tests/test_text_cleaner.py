
from __future__ import annotations

import pandas as pd

from src.preprocessing.text_cleaner import clean_text, preprocess_corpus


def test_clean_text_handles_empty_string():
    assert clean_text("") == []


def test_clean_text_removes_domain_stopwords():
    tokens = clean_text("This study shows important results from the analysis", use_spacy=False)
    assert "study" not in tokens
    assert "important" not in tokens
    assert "analysis" not in tokens


def test_clean_text_removes_digits_and_short_tokens():
    tokens = clean_text("AI in 2024 improves x diagnosis", use_spacy=False)
    assert "2024" not in tokens
    assert "x" not in tokens


def test_preprocess_corpus_creates_expected_columns():
    df = pd.DataFrame(
        {
            "title": ["AI diagnosis"],
            "abstract": ["Machine learning improves diagnosis and treatment in clinical care."],
            "has_abstract": [True],
        }
    )
    result = preprocess_corpus(df, use_spacy=False, min_token_threshold=2)
    assert "tokens" in result.columns
    assert "cleaned_text" in result.columns
    assert "token_count" in result.columns
    assert "eligible_for_text_analysis" in result.columns


def test_short_documents_are_flagged_out_of_text_analysis():
    df = pd.DataFrame(
        {
            "title": ["Letter"],
            "abstract": ["Brief note."],
            "has_abstract": [True],
        }
    )
    result = preprocess_corpus(df, use_spacy=False, min_token_threshold=20)
    assert result.loc[0, "eligible_for_text_analysis"] is False or result.loc[0, "eligible_for_text_analysis"] == False
