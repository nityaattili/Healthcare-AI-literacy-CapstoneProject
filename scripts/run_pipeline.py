#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEMO_DIR,
    OUTPUT_DIR,
    get_paths,
    N_TOPICS,
    KEYWORD_TOP_N,
    COCITATION_MIN_COUNT,
    COCITATION_TOP_N,
)
from src.data_collection import load_papers
from src.preprocessing import preprocess_corpus
from src.nlp import fit_lda, get_topic_summary, extract_keywords, keyword_trends_over_time
from src.nlp.topic_model import save_lda
from src.analytics import author_stats, journal_stats, build_cocitation_edges, cocitation_network


def main():
    get_paths()
    papers_path = DEMO_DIR / "papers.csv"
    if not papers_path.exists():
        print("No data/sample_data/papers.csv. Run: python scripts/download_demo_data.py")
        return 1

    print("Loading papers...")
    df = load_papers(papers_path)
    if df.empty or len(df) < 10:
        print("Too few papers. Run download_demo_data.py first.")
        return 1

    print("Preprocessing...")
    df = preprocess_corpus(df, use_spacy=False)  # NLTK fallback if no spaCy model
    tokens_list = df["tokens"].tolist()

    # LDA
    print("Fitting LDA...")
    model, dictionary, corpus = fit_lda(tokens_list, num_topics=N_TOPICS)
    lda_dir = OUTPUT_DIR / "lda"
    save_lda(model, dictionary, lda_dir)
    topic_summary = get_topic_summary(model, num_words=10)
    topics_data = [
        {"topic_id": tid, "words": [w for w, _ in words]}
        for tid, words in topic_summary
    ]
    with open(OUTPUT_DIR / "topics.json", "w") as f:
        json.dump(topics_data, f, indent=2)

    # Keywords
    print("Keyword extraction...")
    texts = df["cleaned_text"].fillna("").tolist()
    keywords = extract_keywords(texts, top_n=KEYWORD_TOP_N)
    with open(OUTPUT_DIR / "keywords.json", "w") as f:
        json.dump([{"keyword": k, "score": s} for k, s in keywords], f, indent=2)
    kw_trends = keyword_trends_over_time(df, top_n=15)
    kw_trends.to_csv(OUTPUT_DIR / "keyword_trends.csv", index=False)

    # Author & journal stats
    print("Author/journal stats...")
    author_stats(df).to_csv(OUTPUT_DIR / "author_stats.csv", index=False)
    journal_stats(df).to_csv(OUTPUT_DIR / "journal_stats.csv", index=False)
    from src.analytics.author_stats import year_stats
    year_stats(df).to_csv(OUTPUT_DIR / "year_stats.csv", index=False)

    # Co-citation (token-overlap surrogate)
    print("Co-citation network...")
    edges = build_cocitation_edges(
        df,
        min_cooccur=COCITATION_MIN_COUNT,
        top_n_edges=COCITATION_TOP_N,
    )
    G = cocitation_network(edges, node_labels=df, id_col="pmid", title_col="title")
    # Save edges for app
    edges_df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    edges_df.to_csv(OUTPUT_DIR / "cocitation_edges.csv", index=False)
    # Node list with labels
    node_labels = df[["pmid", "title"]].drop_duplicates("pmid")
    node_labels.to_csv(OUTPUT_DIR / "cocitation_nodes.csv", index=False)

    # Save preprocessed corpus for app (sample to avoid huge file)
    df_out = df.drop(columns=["tokens"], errors="ignore")
    df_out.to_csv(OUTPUT_DIR / "papers_processed.csv", index=False)

    print("Pipeline done. Outputs in", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
