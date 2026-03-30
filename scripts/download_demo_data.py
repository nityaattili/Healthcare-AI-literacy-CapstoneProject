#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEMO_DIR, DEMO_QUERY, DEMO_SIZE, get_paths
from src.data_collection import fetch_pubmed_papers, save_papers


def main():
    get_paths()
    out_path = DEMO_DIR / "papers.csv"
    print(f"Fetching up to {DEMO_SIZE} papers for query: {DEMO_QUERY}")
    df = fetch_pubmed_papers(DEMO_QUERY, max_results=DEMO_SIZE, out_path=str(out_path))
    print(f"Saved {len(df)} papers to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
