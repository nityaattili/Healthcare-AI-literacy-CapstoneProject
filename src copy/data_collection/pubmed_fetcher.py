"""Robust PubMed fetching utilities for Meeting 4 data ingestion.

This module supports:
- batched esearch + efetch collection
- rate limiting and optional API key usage
- resilient XML parsing across inconsistent PubMed records
- multi-query collection with PMID deduplication
- metadata flags for abstract availability and source query tracking
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

import pandas as pd

from config.settings import (
    NCBI_API_KEY,
    NCBI_EMAIL,
    PUBMED_BATCH_DELAY_SECONDS,
    PUBMED_BATCH_SIZE,
)

try:
    from Bio import Entrez
except ImportError:  # pragma: no cover
    Entrez = None

LOGGER = logging.getLogger(__name__)
UNIFIED_COLUMNS = [
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


def _ensure_entrez() -> None:
    if Entrez is None:
        raise ImportError("Install biopython to use PubMed fetching: pip install biopython")
    Entrez.email = NCBI_EMAIL
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY


def _safe_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part and part.strip()).strip()


def _parse_abstract(article_element: ET.Element) -> str:
    abstract_parent = article_element.find(".//Abstract")
    if abstract_parent is None:
        return ""

    parts: list[str] = []
    for node in abstract_parent.findall("AbstractText"):
        label = (node.attrib.get("Label") or "").strip()
        text = _safe_text(node)
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)

    if not parts:
        return _safe_text(abstract_parent)
    return " ".join(parts).strip()


def _parse_keywords(medline_citation: ET.Element) -> str:
    values: list[str] = []
    for path in [".//KeywordList/Keyword", ".//MeshHeadingList/MeshHeading/DescriptorName"]:
        for node in medline_citation.findall(path):
            text = _safe_text(node)
            if text:
                values.append(text)

    deduped = list(dict.fromkeys(values))
    return "; ".join(deduped)


def _parse_authors(article_element: ET.Element) -> str:
    authors: list[str] = []
    for author in article_element.findall(".//AuthorList/Author"):
        collective = _safe_text(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue

        fore = _safe_text(author.find("ForeName"))
        last = _safe_text(author.find("LastName"))
        if fore and last:
            authors.append(f"{fore} {last}")
        elif last:
            authors.append(last)
    return "; ".join(authors)


def _parse_year(medline_citation: ET.Element) -> str:
    for path in [
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//DateCompleted/Year",
        ".//DateRevised/Year",
    ]:
        node = medline_citation.find(path)
        text = _safe_text(node)
        if text:
            return text
    return ""


def _parse_pubmed_article(article: ET.Element, source_query: str) -> Optional[dict]:
    """Extract a normalized paper record from a PubMedArticle XML element."""
    try:
        medline_citation = article.find(".//MedlineCitation")
        article_element = article.find(".//MedlineCitation/Article")
        if medline_citation is None or article_element is None:
            return None

        pmid = _safe_text(medline_citation.find("PMID"))
        if not pmid:
            return None

        title = _safe_text(article_element.find("ArticleTitle"))
        abstract = _parse_abstract(article_element)
        authors = _parse_authors(article_element)
        journal = _safe_text(article_element.find(".//Journal/Title"))
        year = _parse_year(medline_citation)
        keywords = _parse_keywords(medline_citation)

        has_abstract = len(abstract.strip()) >= 50
        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
            "keywords": keywords,
            "source_query": source_query,
            "has_abstract": has_abstract,
            "abstract_length": len(abstract.strip()),
        }
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to parse PubMed article: %s", exc)
        return None


def _esearch_ids(query: str, max_results: int) -> list[str]:
    _ensure_entrez()
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_pubmed_batch(
    query: str,
    max_results: int = 500,
    batch_size: int = PUBMED_BATCH_SIZE,
    batch_delay: float = PUBMED_BATCH_DELAY_SECONDS,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch a single PubMed query in batches and return a normalized DataFrame."""
    id_list = _esearch_ids(query=query, max_results=max_results)
    if not id_list:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    rows: list[dict] = []
    for start in range(0, len(id_list), batch_size):
        batch_ids = id_list[start : start + batch_size]
        for attempt in range(1, retries + 1):
            try:
                handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="xml", retmode="xml")
                tree = ET.parse(handle)
                handle.close()
                root = tree.getroot()
                for article in root.findall(".//PubmedArticle"):
                    row = _parse_pubmed_article(article, source_query=query)
                    if row:
                        rows.append(row)
                break
            except Exception as exc:  # pragma: no cover
                LOGGER.warning(
                    "Batch fetch failed for query=%s batch_start=%s attempt=%s: %s",
                    query,
                    start,
                    attempt,
                    exc,
                )
                if attempt == retries:
                    LOGGER.error("Giving up on batch starting at %s for query=%s", start, query)
                else:
                    time.sleep(min(2**attempt, 10))
        time.sleep(batch_delay)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    return frame.reindex(columns=UNIFIED_COLUMNS)


def fetch_multiple_queries(
    query_map: dict[str, str],
    max_results: int,
    batch_size: int = PUBMED_BATCH_SIZE,
    batch_delay: float = PUBMED_BATCH_DELAY_SECONDS,
) -> pd.DataFrame:
    """Run multiple PubMed queries, merge results, and deduplicate on PMID.

    The first-seen query wins for source_query metadata, matching the advisor note
    recorded in Meeting 4.
    """
    frames: list[pd.DataFrame] = []
    for _, query in query_map.items():
        df = fetch_pubmed_batch(
            query=query,
            max_results=max_results,
            batch_size=batch_size,
            batch_delay=batch_delay,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["pmid"], keep="first").reset_index(drop=True)
    return combined.reindex(columns=UNIFIED_COLUMNS)


def save_fetch_outputs(df: pd.DataFrame, csv_path: str, json_path: Optional[str] = None) -> None:
    """Persist collected papers in CSV and optionally JSON."""
    if df.empty:
        return
    pd.DataFrame(df).to_csv(csv_path, index=False)
    if json_path:
        pd.DataFrame(df).to_json(json_path, orient="records", indent=2)
