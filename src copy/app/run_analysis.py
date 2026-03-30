"""Run the analysis pipeline and return structured outputs for the app."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_config():
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from config import (
        COCITATION_MIN_COUNT,
        COCITATION_TOP_N,
        KEYWORD_TOP_N,
        MIN_TOKEN_THRESHOLD,
        N_TOPICS,
    )

    return N_TOPICS, KEYWORD_TOP_N, COCITATION_MIN_COUNT, COCITATION_TOP_N, MIN_TOKEN_THRESHOLD


def run_pipeline_on_dataframe(
    df: pd.DataFrame,
    n_topics: Optional[int] = None,
    keyword_top_n: Optional[int] = None,
    cocitation_min: Optional[int] = None,
    cocitation_top_n: Optional[int] = None,
    use_spacy: bool = False,
) -> Dict[str, Any]:
    if df.empty or len(df) < 5:
        return _empty_results()

    try:
        (
            default_topics,
            default_keywords,
            default_cocitation_min,
            default_cocitation_top,
            min_token_threshold,
        ) = _get_config()
    except Exception:
        default_topics, default_keywords = 8, 50
        default_cocitation_min, default_cocitation_top = 2, 100
        min_token_threshold = 20

    n_topics = n_topics or default_topics
    keyword_top_n = keyword_top_n or default_keywords
    cocitation_min = cocitation_min or default_cocitation_min
    cocitation_top_n = cocitation_top_n or default_cocitation_top

    from src.analytics import author_stats, build_cocitation_edges, journal_stats
    from src.analytics.author_stats import year_stats
    from src.nlp import extract_keywords, fit_lda, get_topic_summary, keyword_trends_over_time
    from src.preprocessing import preprocess_corpus

    processed_df = preprocess_corpus(
        df.copy(),
        use_spacy=use_spacy,
        min_token_threshold=min_token_threshold,
    )

    text_df = processed_df[processed_df["eligible_for_text_analysis"]].copy()
    if text_df.empty or len(text_df) < 5:
        results = _empty_results()
        results["papers_df"] = processed_df.drop(columns=["tokens"], errors="ignore")
        results["excluded_from_text_analysis"] = int((~processed_df["eligible_for_text_analysis"]).sum())
        return results

    tokens_list = text_df["tokens"].tolist()
    model, dictionary, corpus = fit_lda(
        tokens_list,
        num_topics=min(n_topics, max(2, len(text_df) // 10)),
        passes=5,
    )
    topic_summary = get_topic_summary(model, num_words=10)
    topics_data: List[dict] = [
        {"topic_id": topic_id, "words": [word for word, _ in words]}
        for topic_id, words in topic_summary
    ]

    keywords = extract_keywords(text_df["cleaned_text"].fillna("").tolist(), top_n=keyword_top_n)
    keywords_data = [{"keyword": keyword, "score": float(score)} for keyword, score in keywords]
    keyword_trends = keyword_trends_over_time(text_df, top_n=15)

    author_stats_df = author_stats(processed_df)
    journal_stats_df = journal_stats(processed_df)
    year_stats_df = year_stats(processed_df)

    edges = build_cocitation_edges(
        text_df,
        min_cooccur=cocitation_min,
        top_n_edges=cocitation_top_n,
    )
    cocitation_edges = pd.DataFrame(edges, columns=["source", "target", "weight"])

    papers_df = processed_df.drop(columns=["tokens"], errors="ignore")
    return {
        "topics_data": topics_data,
        "keywords_data": keywords_data,
        "keyword_trends": keyword_trends,
        "author_stats_df": author_stats_df,
        "journal_stats_df": journal_stats_df,
        "year_stats_df": year_stats_df,
        "cocitation_edges": cocitation_edges,
        "papers_df": papers_df,
        "excluded_from_text_analysis": int((~processed_df["eligible_for_text_analysis"]).sum()),
    }


def _empty_results() -> Dict[str, Any]:
    return {
        "topics_data": [],
        "keywords_data": [],
        "keyword_trends": pd.DataFrame(),
        "author_stats_df": pd.DataFrame(),
        "journal_stats_df": pd.DataFrame(),
        "year_stats_df": pd.DataFrame(),
        "cocitation_edges": pd.DataFrame(),
        "papers_df": pd.DataFrame(),
        "excluded_from_text_analysis": 0,
    }
