"""
AI Literacy – Clinical Literature Analysis
Browser-based app: upload or select healthcare papers, explore topic summaries,
keyword trends, author/journal statistics, co-citation networks, and Q&A.
"""
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on path so "src" imports work (e.g. streamlit run src/app/main.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

# Project paths
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
DEMO_DIR = DATA_DIR / "sample_data"

# Chart/table width (Streamlit: use width= for plotly/dataframe)
CHART_WIDTH = "stretch"


def pubmed_url(pmid) -> Optional[str]:
    """Return PubMed article URL if pmid looks like a numeric ID; else None."""
    if pd.isna(pmid):
        return None
    s = str(pmid).strip()
    if not s:
        return None
    if s.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{s}/"
    return None


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def document_download_data(row: pd.Series, fmt: str = "csv") -> str:
    """Return string content to download one document as CSV or JSON."""
    df_one = pd.DataFrame([row]).drop(columns=["_relevance", "_search_text"], errors="ignore")
    if fmt == "json":
        return df_one.to_json(orient="records", indent=2)
    return df_one.to_csv(index=False)


def load_demo_results():
    """Load pre-computed pipeline results from output/ (demo dataset)."""
    if not (OUTPUT_DIR / "topics.json").exists():
        return None
    return {
        "topics_data": load_json(OUTPUT_DIR / "topics.json") or [],
        "keywords_data": load_json(OUTPUT_DIR / "keywords.json") or [],
        "keyword_trends": load_csv(OUTPUT_DIR / "keyword_trends.csv"),
        "author_stats_df": load_csv(OUTPUT_DIR / "author_stats.csv"),
        "journal_stats_df": load_csv(OUTPUT_DIR / "journal_stats.csv"),
        "year_stats_df": load_csv(OUTPUT_DIR / "year_stats.csv"),
        "cocitation_edges": load_csv(OUTPUT_DIR / "cocitation_edges.csv"),
        "papers_df": load_csv(OUTPUT_DIR / "papers_processed.csv"),
    }


def main():
    st.set_page_config(page_title="AI Literacy – Literature Analysis", layout="wide")
    st.title("AI Literacy – Clinical Literature Analysis")
    st.caption(
        "Upload or select healthcare papers, then explore topic summaries, "
        "keyword trends, author/journal statistics, and co-citation networks."
    )

    # ----- Sidebar: data source -----
    st.sidebar.header("Data source")
    # Full sample CSV download
    from src.app.sample_data import get_full_sample_df
    full_sample = get_full_sample_df()
    st.sidebar.download_button(
        "Download full sample CSV",
        data=full_sample.to_csv(index=False),
        file_name="sample_healthcare_papers.csv",
        mime="text/csv",
        help="Complete sample dataset with 25 healthcare papers (title, abstract, authors, journal, year, pmid, keywords).",
    )
    st.sidebar.divider()
    data_source = st.sidebar.radio(
        "Choose how to load papers",
        ["Use demo dataset (pre-computed)", "Upload my papers (CSV or JSON)"],
        label_visibility="collapsed",
    )

    results = None
    source_label = ""

    if data_source == "Use demo dataset (pre-computed)":
        results = load_demo_results()
        source_label = "Demo dataset"
        if results is None:
            st.warning(
                "Demo pipeline outputs not found. Either run the pipeline first:\n\n"
                "1. `python scripts/generate_demo_csv.py`\n"
                "2. `python scripts/run_pipeline.py`\n\n"
                "Or **upload your own papers** using the sidebar."
            )
            st.stop()

    else:
        # Upload
        uploaded = st.sidebar.file_uploader(
            "Upload healthcare papers (CSV or JSON)",
            type=["csv", "json"],
            help="CSV/JSON with columns: title, abstract, authors, journal, year; pmid or id optional.",
        )
        # Use existing results from session if we have them (e.g. after rerun when file is no longer selected)
        if "upload_results" in st.session_state:
            results = st.session_state["upload_results"]
            source_label = f"Uploaded ({st.session_state.get('upload_n', 0)} papers)"
            if st.sidebar.button("Clear and upload a different file"):
                del st.session_state["upload_results"]
                if "upload_n" in st.session_state:
                    del st.session_state["upload_n"]
                st.rerun()
        elif uploaded is not None:
            if st.sidebar.button("Run analysis", type="primary"):
                with st.spinner("Preprocessing and running analysis…"):
                    from src.data_collection.io_utils import load_papers_from_bytes
                    from src.app.run_analysis import run_pipeline_on_dataframe
                    data = uploaded.getvalue()
                    df = load_papers_from_bytes(data, uploaded.name)
                    if len(df) < 5:
                        st.sidebar.error("Need at least 5 papers. Check your file columns.")
                    else:
                        res = run_pipeline_on_dataframe(df)
                        st.session_state["upload_results"] = res
                        st.session_state["upload_n"] = len(df)
                        st.sidebar.success(f"Analyzed {len(df)} papers.")
                        st.rerun()
            # Same run: show message until they click Run
            st.info("Click **Run analysis** in the sidebar to analyze your uploaded file.")
            st.stop()
        else:
            st.info("Use the sidebar to **upload a CSV or JSON** file with healthcare papers, then click **Run analysis**.")
            st.stop()

    if results is None:
        st.stop()

    # ----- Tabs: same views for demo or uploaded -----
    topics_data = results["topics_data"]
    keywords_data = results["keywords_data"]
    keyword_trends = results["keyword_trends"]
    author_stats_df = results["author_stats_df"]
    journal_stats_df = results["journal_stats_df"]
    year_stats_df = results["year_stats_df"]
    cocitation_edges = results["cocitation_edges"]
    papers_df = results["papers_df"]

    st.sidebar.success(f"Showing: **{source_label}**")

    tab1, tab2, tab3, tab4, tab5, tab6, tab_qa = st.tabs([
        "Overview",
        "Topic summaries",
        "Keyword trends",
        "Author & journal stats",
        "Co-citation network",
        "Data",
        "Q&A",
    ])

    with tab1:
        st.subheader("Overview")
        if not year_stats_df.empty:
            st.bar_chart(year_stats_df.set_index("year"))
        st.metric("Papers in corpus", len(papers_df) if not papers_df.empty else 0)
        if topics_data:
            st.metric("Topics (LDA)", len(topics_data))

    with tab2:
        st.subheader("Topic summaries (LDA)")
        if topics_data:
            for t in topics_data:
                tid = t.get("topic_id", 0)
                words = t.get("words", [])
                st.markdown(f"**Topic {tid}:** " + ", ".join(words))
        else:
            st.info("No topic data.")

    with tab3:
        st.subheader("Keyword trends over time")
        if not keyword_trends.empty and "year" in keyword_trends.columns:
            import plotly.express as px
            top_kw = keyword_trends.groupby("keyword")["score"].sum().nlargest(15).index
            df_plot = keyword_trends[keyword_trends["keyword"].isin(top_kw)]
            fig = px.line(df_plot, x="year", y="score", color="keyword", title="Keyword scores by year")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Top keywords (overall)")
        if keywords_data:
            kw_df = pd.DataFrame(keywords_data).head(30)
            st.dataframe(kw_df, use_container_width=True)

    with tab4:
        st.subheader("Author statistics")
        if not author_stats_df.empty:
            st.dataframe(author_stats_df.head(30), use_container_width=True)
        st.subheader("Journal statistics")
        if not journal_stats_df.empty:
            st.dataframe(journal_stats_df.head(30), use_container_width=True)

    with tab5:
        st.subheader("Co-citation network (related papers by content similarity)")
        if not cocitation_edges.empty and len(cocitation_edges) > 0:
            try:
                import networkx as nx
                import plotly.graph_objects as go
                G = nx.from_pandas_edgelist(
                    cocitation_edges,
                    source="source",
                    target="target",
                    edge_attr="weight",
                )
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
                edge_x, edge_y = [], []
                for e in G.edges():
                    x0, y0 = pos[e[0]]
                    x1, y1 = pos[e[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                node_x = [pos[n][0] for n in G.nodes()]
                node_y = [pos[n][1] for n in G.nodes()]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#888")))
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode="markers",
                    marker=dict(size=8, color="steelblue"),
                    text=list(G.nodes()),
                    hoverinfo="text",
                ))
                fig.update_layout(showlegend=False, margin=dict(b=0, l=0, r=0, t=30))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.code(str(e))
                st.dataframe(cocitation_edges.head(50))
        else:
            st.info("No co-citation edges. More papers may be needed for a visible network.")

    with tab6:
        st.subheader("Processed papers")
        if not papers_df.empty:
            st.download_button(
                "Download all as CSV",
                data=papers_df.drop(columns=["_relevance", "_search_text"], errors="ignore").to_csv(index=False),
                file_name="healthcare_papers.csv",
                mime="text/csv",
                key="dl_all_papers",
            )
            st.markdown("---")
            st.markdown("**Select a document to view details, link, and download.**")
            display_df = papers_df.drop(columns=["_relevance", "_search_text"], errors="ignore")
            titles = display_df["title"].astype(str) if "title" in display_df.columns else display_df.index.astype(str)
            titles_list = titles.tolist()[:200]
            selected_title = st.selectbox("Document", options=titles_list, index=0, label_visibility="collapsed")
            if selected_title is not None:
                row = display_df[display_df["title"].astype(str) == selected_title].iloc[0]
                with st.container():
                    st.markdown(f"### {row.get('title', '')}")
                    st.caption(f"**Authors:** {row.get('authors', '')}  |  **Journal:** {row.get('journal', '')}  |  **Year:** {row.get('year', '')}")
                    if pd.notna(row.get("abstract")):
                        st.write(row["abstract"][:500] + ("…" if len(str(row.get("abstract", ""))) > 500 else ""))
                    pmid = row.get("pmid")
                    url = pubmed_url(pmid)
                    col1, col2 = st.columns(2)
                    with col1:
                        if url:
                            st.link_button("View on PubMed", url)
                        else:
                            st.caption("(PubMed link available for numeric PMID only)")
                    with col2:
                        st.download_button("Download this document (CSV)", data=document_download_data(row), file_name=f"paper_{pmid}.csv", mime="text/csv", key=f"dl_doc_{pmid}")
        else:
            st.info("No papers loaded.")

    with tab_qa:
        st.subheader("Q&A — Search the literature")
        st.caption("Ask a question or enter keywords to find relevant papers. Results show document links and allow download.")
        query_qa = st.text_input("Question or search terms", placeholder="e.g. machine learning diagnosis accuracy", label_visibility="collapsed")
        if query_qa and query_qa.strip():
            from src.app.search_qa import search_papers, snippet
            ranked, scores = search_papers(query_qa, papers_df.drop(columns=["_relevance", "_search_text"], errors="ignore"), top_k=15)
            if ranked.empty:
                st.info("No matching papers.")
            else:
                st.write(f"**{len(ranked)}** relevant paper(s):")
                for idx, (_, row) in enumerate(ranked.iterrows()):
                    rel = scores[idx] if idx < len(scores) else 0
                    with st.expander(f"**{row.get('title', 'N/A')}** (relevance: {rel:.2f})"):
                        st.caption(f"Authors: {row.get('authors', '')}  |  {row.get('journal', '')} ({row.get('year', '')})")
                        st.write(snippet(str(row.get("abstract", "")), query_qa, max_len=300))
                        pmid = row.get("pmid")
                        url = pubmed_url(pmid)
                        c1, c2 = st.columns(2)
                        with c1:
                            if url:
                                st.link_button("View on PubMed", url)
                            else:
                                st.caption("(No PubMed link for this ID)")
                        with c2:
                            st.download_button("Download document", data=document_download_data(row), file_name=f"paper_{pmid}.csv", mime="text/csv", key=f"dl_qa_{idx}")
        else:
            st.info("Enter a question or search terms above to find relevant papers.")


if __name__ == "__main__":
    main()
