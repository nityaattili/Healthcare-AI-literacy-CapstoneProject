"""
LDA topic modeling on tokenized corpus (gensim).
"""
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel


def fit_lda(
    documents: List[List[str]],
    num_topics: int = 8,
    passes: int = 10,
    random_state: int = 42,
    min_df: int = 2,
) -> Tuple[LdaModel, corpora.Dictionary, List[Tuple[int, float]]]:
    """
    Fit LDA on list of token lists. Returns (model, dictionary, corpus).
    corpus is list of (id, count) bags.
    """
    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=min_df, no_above=0.95, keep_n=5000)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    model = LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        random_state=random_state,
        alpha="auto",
        eta="auto",
    )
    return model, dictionary, corpus


def get_topic_summary(
    model: LdaModel,
    num_words: int = 10,
) -> List[Tuple[int, List[Tuple[str, float]]]]:
    """Return list of (topic_id, [(word, weight), ...]) for each topic."""
    return [
        (tid, model.show_topic(tid, topn=num_words))
        for tid in range(model.num_topics)
    ]


def get_document_topics(
    model: LdaModel,
    corpus: List[List[Tuple[int, int]]],
) -> List[List[Tuple[int, float]]]:
    """Return per-document topic distribution for each doc in corpus."""
    return [model.get_document_topics(bow) for bow in corpus]


def save_lda(model: LdaModel, dictionary: corpora.Dictionary, path: Path) -> None:
    """Save model and dictionary to directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path / "lda.model"))
    dictionary.save(str(path / "lda.dict"))


def load_lda(path: Path) -> Tuple[LdaModel, corpora.Dictionary]:
    """Load model and dictionary from directory."""
    path = Path(path)
    model = LdaModel.load(str(path / "lda.model"))
    dictionary = corpora.Dictionary.load(str(path / "lda.dict"))
    return model, dictionary


if __name__ == "__main__":
    docs = [
        ["machine", "learning", "healthcare", "diagnosis"],
        ["deep", "learning", "clinical", "data"],
        ["patient", "health", "treatment", "care"],
    ] * 5
    m, d, c = fit_lda(docs, num_topics=2)
    for tid, words in get_topic_summary(m, num_words=5):
        print(f"Topic {tid}:", [w for w, _ in words])
