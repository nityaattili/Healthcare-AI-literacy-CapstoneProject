"""
Keyword extraction and temporal trends (sklearn CountVectorizer / TF-IDF).
"""
from collections import defaultdict
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def extract_keywords(
    texts: List[str],
    top_n: int = 50,
    min_df: int = 2,
    max_df: float = 0.95,
    use_tfidf: bool = True,
) -> List[tuple]:
    """
    Extract top_n keywords from a list of documents.
    Returns list of (term, score) sorted by score descending.
    """
    if not texts:
        return []
    if use_tfidf:
        vec = TfidfVectorizer(max_features=top_n * 3, min_df=min_df, max_df=max_df)
        X = vec.fit_transform(texts)
        scores = X.sum(axis=0).A1
        idx = scores.argsort()[::-1][:top_n]
        return [(vec.get_feature_names_out()[i], float(scores[i])) for i in idx]
    else:
        vec = CountVectorizer(max_features=top_n * 3, min_df=min_df, max_df=max_df)
        X = vec.fit_transform(texts)
        scores = X.sum(axis=0).A1
        idx = scores.argsort()[::-1][:top_n]
        return [(vec.get_feature_names_out()[i], float(scores[i])) for i in idx]


def keyword_trends_over_time(
    df: pd.DataFrame,
    text_column: str = "cleaned_text",
    year_column: str = "year",
    top_n: int = 20,
    min_docs_per_year: int = 5,
) -> pd.DataFrame:
    """
    For each year, compute top keywords; return long-format DataFrame
    (year, keyword, count or rank) for visualization.
    """
    if year_column not in df.columns or text_column not in df.columns:
        return pd.DataFrame(columns=["year", "keyword", "count"])
    df = df[df[year_column].notna()].copy()
    df[year_column] = pd.to_numeric(df[year_column], errors="coerce")
    df = df.dropna(subset=[year_column])
    if df.empty:
        return pd.DataFrame(columns=["year", "keyword", "count"])
    rows = []
    for year, g in df.groupby(year_column):
        if len(g) < min_docs_per_year:
            continue
        texts = g[text_column].fillna("").tolist()
        kw = extract_keywords(texts, top_n=top_n, use_tfidf=True)
        for keyword, score in kw:
            rows.append({"year": int(year), "keyword": keyword, "score": score})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    texts = ["machine learning in health", "health care and machine learning"] * 3
    print(extract_keywords(texts, top_n=5))
