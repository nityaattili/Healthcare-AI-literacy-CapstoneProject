#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_paths, DEMO_DIR

SAMPLE_ABSTRACTS = [
    "Machine learning in healthcare improves diagnosis and treatment outcomes.",
    "Deep learning models for medical image analysis have shown high accuracy.",
    "Natural language processing enables extraction of clinical information from notes.",
    "AI literacy among clinicians and patients is essential for adoption.",
    "Ethical considerations in artificial intelligence for healthcare.",
    "Predictive analytics support clinical decision making.",
    "Electronic health records and machine learning for risk stratification.",
    "Patient perception of AI in medicine and trust in algorithms.",
    "Barriers to clinical adoption of artificial intelligence tools.",
    "Systematic review of AI applications in radiology and pathology.",
]


def main():
    get_paths()
    import pandas as pd
    n = 80  # small set for quick run
    rows = []
    for i in range(n):
        rows.append({
            "pmid": f"DEMO{i:05d}",
            "title": f"Demo paper on AI in healthcare {i}",
            "abstract": SAMPLE_ABSTRACTS[i % len(SAMPLE_ABSTRACTS)],
            "authors": f"Author A{i % 10}; Author B{i % 5}",
            "journal": ["Nature", "JAMA", "Lancet", "BMJ", "PLOS One"][i % 5],
            "year": 2020 + (i % 5),
            "keywords": "machine learning; healthcare; AI",
        })
    df = pd.DataFrame(rows)
    out = DEMO_DIR / "papers.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
