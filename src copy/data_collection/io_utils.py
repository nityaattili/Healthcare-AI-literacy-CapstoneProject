"""I/O utilities for normalized paper datasets."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pandas as pd

NORMALIZED_COLUMNS = [
    "pmid",
    "title",
    "abstract",
    "authors",
    "journal",
    "year",
    "keywords",
    "source_query",
    "has_abstract",
    "abstract_length",
]


def _normalize_papers_df(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns and "pmid" not in df.columns:
        df = df.rename(columns={"id": "pmid"})

    for column in NORMALIZED_COLUMNS:
        if column not in df.columns:
            if column == "has_abstract":
                df[column] = False
            elif column == "abstract_length":
                df[column] = 0
            else:
                df[column] = ""

    df["pmid"] = df["pmid"].fillna("").astype(str)
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["authors"] = df["authors"].fillna("")
    df["journal"] = df["journal"].fillna("")
    df["year"] = df["year"].fillna("").astype(str)
    df["keywords"] = df["keywords"].fillna("")
    df["source_query"] = df["source_query"].fillna("")
    df["has_abstract"] = df["has_abstract"].fillna(False).astype(bool)
    df["abstract_length"] = (
        pd.to_numeric(df["abstract_length"], errors="coerce")
        .fillna(df["abstract"].astype(str).str.len())
        .astype(int)
    )
    return df.reindex(columns=NORMALIZED_COLUMNS)


def load_papers_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(data))
    elif suffix == ".json":
        df = pd.read_json(io.BytesIO(data))
    elif suffix == ".jsonl":
        df = pd.read_json(io.BytesIO(data), lines=True)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use CSV, JSON, or JSONL.")
    return _normalize_papers_df(df)


def load_papers(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
    elif path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    return _normalize_papers_df(df)


def save_papers(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_papers_df(df.copy())

    if path.suffix.lower() == ".json":
        normalized.to_json(path, orient="records", indent=2)
    elif path.suffix.lower() == ".jsonl":
        normalized.to_json(path, orient="records", lines=True)
    else:
        normalized.to_csv(path, index=False)
