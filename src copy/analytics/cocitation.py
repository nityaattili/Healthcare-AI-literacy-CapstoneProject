
from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
import networkx as nx


def build_cocitation_edges(
    df: pd.DataFrame,
    id_column: str = "pmid",
    min_cooccur: int = 2,
    top_n_edges: int = 200,
    token_column: str = "tokens",
) -> List[Tuple[str, str, int]]:
    """
    Build edges from paper similarity (e.g. shared tokens) when real citation
    data is not available. Each edge is (id1, id2, weight).
    """
    if id_column not in df.columns or token_column not in df.columns:
        return []
    # Use token overlap as proxy for "relatedness" (co-citation surrogate)
    ids = df[id_column].astype(str).tolist()
    tokens_list = df[token_column].apply(lambda x: set(x) if isinstance(x, (list, set)) else set()).tolist()
    pair_counts = defaultdict(int)
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, min(i + 50, n)):  # limit pairs for speed
            overlap = len(tokens_list[i] & tokens_list[j]) if tokens_list[i] and tokens_list[j] else 0
            if overlap >= min_cooccur:
                pair_counts[(ids[i], ids[j])] = overlap
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:top_n_edges]
    return [(a, b, w) for (a, b), w in sorted_pairs]


def cocitation_network(
    edges: List[Tuple[str, str, int]],
    node_labels: Optional[pd.DataFrame] = None,
    id_col: str = "pmid",
    title_col: str = "title",
) -> nx.Graph:
    """
    Build NetworkX graph from edges. Optionally attach labels from node_labels DataFrame.
    """
    G = nx.Graph()
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)
    if node_labels is not None and id_col in node_labels.columns:
        titles = node_labels.set_index(id_col)[title_col].to_dict() if title_col in node_labels.columns else {}
        for n in G.nodes():
            G.nodes[n]["label"] = titles.get(str(n), str(n)[:30])
    return G


def most_cited_list(
    df: pd.DataFrame,
    id_column: str = "pmid",
    title_column: str = "title",
    top_n: int = 50,
) -> pd.DataFrame:
    """
    When we don't have citation counts, return top papers by some proxy
    (e.g. word count in abstract). For real data, replace with actual citation count.
    """
    if title_column not in df.columns:
        return pd.DataFrame()
    out = df[[id_column, title_column]].drop_duplicates().head(top_n)
    out = out.rename(columns={id_column: "pmid", title_column: "title"})
    return out
