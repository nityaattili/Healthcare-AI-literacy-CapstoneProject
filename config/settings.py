"""Central project settings for the AI Literacy literature analysis project.

This module keeps environment-specific settings, data collection parameters,
preprocessing options, and search/indexing paths in one place so the pipeline
can be tuned without editing multiple source files.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DEMO_DIR = DATA_DIR / "demo"
CHROMA_PERSIST_DIR = OUTPUT_DIR / "chromadb"

# Collection / retrieval
COLLECTION_NAME = "ai_literacy_healthcare"
DEMO_QUERY = "AI healthcare machine learning clinical"
DEMO_SIZE = 500

# Topic / keyword analysis
N_TOPICS = 8
KEYWORD_TOP_N = 50
MIN_DF = 2
MAX_DF_RATIO = 0.95
MIN_TOKEN_THRESHOLD = 20

# Co-citation
COCITATION_MIN_COUNT = 2
COCITATION_TOP_N = 100

# PubMed collection plan from advisor-approved Meeting 4 scope
PUBMED_QUERY_THEMES = {
    "ai_literacy": '"AI literacy healthcare"',
    "patient_perception": '"patient perception artificial intelligence medicine"',
    "adoption_barriers": '"barriers clinical adoption artificial intelligence"',
}
PUBMED_MAX_PER_QUERY = 200
PUBMED_BATCH_SIZE = 100
PUBMED_BATCH_DELAY_SECONDS = 0.34

# Runtime / API settings
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "ai.literacy@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Preprocessing / stopwords
DOMAIN_STOPWORDS = {
    "study",
    "paper",
    "result",
    "method",
    "analysis",
    "approach",
    "finding",
    "conclusion",
    "data",
    "use",
    "also",
    "may",
    "even",
    "well",
    "however",
    "therefore",
    "show",
    "suggest",
    "include",
    "high",
    "large",
    "significant",
    "important",
}


def get_paths() -> dict[str, Path]:
    """Ensure required directories exist and return them."""
    for path in [DATA_DIR, OUTPUT_DIR, DEMO_DIR, CHROMA_PERSIST_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "data": DATA_DIR,
        "output": OUTPUT_DIR,
        "demo": DEMO_DIR,
        "chroma": CHROMA_PERSIST_DIR,
    }
