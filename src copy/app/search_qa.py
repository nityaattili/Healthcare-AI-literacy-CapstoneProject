"""
Search and Q&A over the papers corpus: rank papers by query (TF-IDF).
"""
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def search_papers(
    query: str,
    papers_df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    top_k: int = 20,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Rank papers by relevance to the query using TF-IDF and cosine similarity.
    Returns (papers_df with rows in rank order, list of relevance scores).
    """
    if papers_df.empty or not query or not query.strip():
        return papers_df.head(0), []

    text_columns = text_columns or ["title", "abstract"]
    available = [c for c in text_columns if c in papers_df.columns]
    if not available:
        return papers_df.head(0), []

    # Combine text for each document
    papers_df = papers_df.copy()
    papers_df["_search_text"] = papers_df[available].fillna("").agg(" ".join, axis=1)
    texts = papers_df["_search_text"].tolist()

    vectorizer = TfidfVectorizer(max_features=5000, min_df=1, stop_words="english")
    try:
        X = vectorizer.fit_transform(texts)
    except Exception:
        return papers_df.drop(columns=["_search_text"], errors="ignore").head(0), []

    q_vec = vectorizer.transform([query.strip()])
    scores = cosine_similarity(q_vec, X).ravel()
    order = scores.argsort()[::-1][:top_k]
    ranked = papers_df.iloc[order].copy()
    ranked["_relevance"] = scores[order]
    ranked = ranked.drop(columns=["_search_text"], errors="ignore")
    return ranked, scores[order].tolist()


def snippet(text: str, query: str, max_len: int = 200) -> str:
    """Return a short snippet of text around the query (or start if no match)."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    q = query.strip().lower()
    if q and q in text.lower():
        pos = text.lower().index(q)
        start = max(0, pos - 50)
        end = min(len(text), pos + max_len - 50)
        excerpt = text[start:end]
        if start > 0:
            excerpt = "… " + excerpt
        if end < len(text):
            excerpt = excerpt + " …"
        return excerpt
    return text[:max_len] + "…"
