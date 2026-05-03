"""
topic_analysis_page.py — Streamlit page for news topic modelling.

Supports:
  - LDA  (always available via scikit-learn)
  - BERTopic (optional, requires bertopic + sentence-transformers)

Data sources: gossipcop_fake/real + politifact_fake/real CSVs from config.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Optional BERTopic ─────────────────────────────────────────────────────────
_BERTOPIC_AVAILABLE = False
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    _BERTOPIC_AVAILABLE = True
except ImportError:
    pass

# ── LDA (always available) ────────────────────────────────────────────────────
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicResult:
    method: str                                      # "lda" | "bertopic"
    n_topics: int
    topics: List[Dict]                               # list of topic dicts
    doc_topic_matrix: Optional[np.ndarray] = None   # (n_docs, n_topics) probs
    doc_assignments: Optional[List[int]] = None      # dominant topic per doc
    coherence_info: str = ""
    error: Optional[str] = None

    @property
    def topic_labels(self) -> List[str]:
        return [t["label"] for t in self.topics]

    @property
    def topic_sizes(self) -> List[int]:
        return [t["size"] for t in self.topics]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS_EXTRA = {
    "say", "said", "says", "new", "year", "years", "day", "days",
    "just", "like", "make", "made", "want", "know", "time", "look",
    "going", "come", "think", "people", "man", "woman", "way",
    "via", "photo", "video", "news", "report", "update", "breaking",
    "read", "watch", "click", "source", "latest", "exclusive",
}


def _clean_for_topic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_data(show_spinner=False)
def _load_dataset(max_per_class: int = 3000) -> pd.DataFrame:
    """Load and cache the four CSV files. Returns a tidy DataFrame."""
    try:
        from config import GOSSIPCOP_FAKE, GOSSIPCOP_REAL, POLITIFACT_FAKE, POLITIFACT_REAL
    except ImportError:
        st.error("Cannot import config.py — make sure you run from the project root.")
        return pd.DataFrame()

    files = [
        (GOSSIPCOP_FAKE, "fake"),
        (GOSSIPCOP_REAL, "real"),
        (POLITIFACT_FAKE, "fake"),
        (POLITIFACT_REAL, "real"),
    ]

    dfs = []
    for path, label in files:
        try:
            df = pd.read_csv(path)
            # pick best text column
            text_col = next(
                (c for c in ["text", "content", "article", "body", "title"] if c in df.columns),
                None,
            )
            if text_col is None:
                continue
            title_col = "title" if "title" in df.columns else None
            sub = pd.DataFrame()
            sub["text"] = df[text_col].fillna("").astype(str)
            sub["title"] = df[title_col].fillna("").astype(str) if title_col else ""
            sub["label"] = label
            sub["source"] = path.rsplit("/", 1)[-1].replace(".csv", "")
            dfs.append(sub.head(max_per_class))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged["clean_text"] = merged["text"].apply(_clean_for_topic)
    merged = merged[merged["clean_text"].str.split().str.len() >= 10].reset_index(drop=True)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# LDA
# ─────────────────────────────────────────────────────────────────────────────

def run_lda(
    texts: List[str],
    n_topics: int = 10,
    n_top_words: int = 10,
) -> TopicResult:
    stop_words_combined = list(
        CountVectorizer(stop_words="english").get_stop_words() | _STOP_WORDS_EXTRA
    )

    vectorizer = CountVectorizer(
        max_features=8000,
        min_df=5,
        max_df=0.85,
        stop_words=stop_words_combined,
        ngram_range=(1, 2),
    )
    try:
        dtm = vectorizer.fit_transform(texts)
    except Exception as exc:
        return TopicResult(method="lda", n_topics=0, topics=[], error=str(exc))

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=15,
        learning_method="online",
        batch_size=256,
    )
    doc_topic_matrix = lda.fit_transform(dtm)
    doc_assignments = doc_topic_matrix.argmax(axis=1).tolist()

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_id, topic_vec in enumerate(lda.components_):
        top_indices = topic_vec.argsort()[: -n_top_words - 1 : -1]
        top_words = [feature_names[i] for i in top_indices]
        label = " · ".join(top_words[:4])
        size = int((doc_topic_matrix.argmax(axis=1) == topic_id).sum())
        docs: List[str] = []
        for doc_idx, assignment in enumerate(doc_assignments):
            if assignment == topic_id and len(docs) < 5:
                docs.append(texts[doc_idx])
        topics.append(
            {
                "id": topic_id,
                "label": label,
                "top_words": top_words,
                "size": size,
                "docs": docs,
            }
        )

    topics = sorted(topics, key=lambda t: t["size"], reverse=True)
    for rank, t in enumerate(topics):
        t["rank"] = rank

    perplexity = lda.perplexity(dtm)
    coherence_info = f"LDA perplexity: {perplexity:.1f} (lower is better)"

    return TopicResult(
        method="lda",
        n_topics=n_topics,
        topics=topics,
        doc_topic_matrix=doc_topic_matrix,
        doc_assignments=doc_assignments,
        coherence_info=coherence_info,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BERTopic
# ─────────────────────────────────────────────────────────────────────────────

def run_bertopic(
    texts: List[str],
    n_topics: int = 10,
    n_top_words: int = 10,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> TopicResult:
    if not _BERTOPIC_AVAILABLE:
        return TopicResult(
            method="bertopic",
            n_topics=0,
            topics=[],
            error="BERTopic is not installed. Run: uv add bertopic sentence-transformers",
        )

    try:
        model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=n_topics,
            top_n_words=n_top_words,
            verbose=False,
        )
        topic_ids, _ = model.fit_transform(texts)

        topic_info = model.get_topic_info()
        topics = []
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:  # outlier cluster
                continue
            top_words_tuples = model.get_topic(tid) or []
            top_words = [w for w, _ in top_words_tuples[:n_top_words]]
            label = " · ".join(top_words[:4]) if top_words else f"Topic {tid}"
            size = int(row.get("Count", 0))
            docs: List[str] = [
                texts[i]
                for i, t in enumerate(topic_ids)
                if t == tid
            ][:5]
            topics.append(
                {
                    "id": tid,
                    "label": label,
                    "top_words": top_words,
                    "size": size,
                    "docs": docs,
                    "rank": 0,
                }
            )

        topics = sorted(topics, key=lambda t: t["size"], reverse=True)
        for rank, t in enumerate(topics):
            t["rank"] = rank

        doc_assignments = [int(t) if t != -1 else -1 for t in topic_ids]

        return TopicResult(
            method="bertopic",
            n_topics=len(topics),
            topics=topics,
            doc_assignments=doc_assignments,
            coherence_info=f"BERTopic — embedding model: {embedding_model}",
        )

    except Exception as exc:
        return TopicResult(method="bertopic", n_topics=0, topics=[], error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# LDA heatmap helper
# ─────────────────────────────────────────────────────────────────────────────

def _doc_topic_heatmap(result: TopicResult, df: pd.DataFrame) -> None:
    if result.doc_topic_matrix is None:
        return

    sample_size = min(60, len(df))
    sample_idx = np.random.choice(len(df), size=sample_size, replace=False)
    matrix_sample = result.doc_topic_matrix[sample_idx]

    topic_labels = [f"T{t['id']}: {t['label'][:20]}" for t in result.topics]
    heatmap_df = pd.DataFrame(
        matrix_sample,
        columns=[f"T{t['id']}" for t in result.topics],
        index=[f"doc{i}" for i in range(sample_size)],
    )
    st.dataframe(
        heatmap_df.style.background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True,
        height=300,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Topic inference for a custom article
# ─────────────────────────────────────────────────────────────────────────────

def _infer_topic_lda(text: str, result: TopicResult) -> Optional[Dict]:
    """
    Very lightweight inference: find the stored topic whose top words overlap
    most with the cleaned input text — no need to re-fit the vectorizer.
    """
    clean = _clean_for_topic(text)
    tokens = set(clean.split())
    best_topic = None
    best_score = -1.0
    for topic in result.topics:
        overlap = len(tokens & set(topic["top_words"]))
        score = overlap / max(len(topic["top_words"]), 1)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic if best_score > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Main page renderer — called from app.py
# ─────────────────────────────────────────────────────────────────────────────

def render_topic_analysis_page() -> None:
    st.markdown(
        "Explore the underlying topics in the GossipCop and PolitiFact datasets "
        "using LDA or BERTopic."
    )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Topic analysis settings")

        method_options = ["LDA"]
        if _BERTOPIC_AVAILABLE:
            method_options.append("BERTopic")
        method = st.selectbox("Method", method_options)

        n_topics = st.slider("Number of topics", min_value=3, max_value=30, value=10, step=1)
        n_top_words = st.slider("Top words per topic", min_value=5, max_value=20, value=10, step=1)

        label_filter = st.multiselect(
            "Filter by label",
            options=["fake", "real"],
            default=["fake", "real"],
        )

        max_docs = st.slider(
            "Max documents to analyse",
            min_value=200,
            max_value=5000,
            value=2000,
            step=200,
        )

        run_button = st.button("Run topic analysis", type="primary")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset…"):
        df_full = _load_dataset()

    if df_full.empty:
        st.error(
            "No dataset files found. Make sure the CSV files exist in the `data/` directory "
            "and run from the project root."
        )
        return

    df_filtered = df_full[df_full["label"].isin(label_filter)].copy()
    if len(df_filtered) > max_docs:
        df_filtered = df_filtered.sample(n=max_docs, random_state=42).reset_index(drop=True)

    st.caption(
        f"Dataset: **{len(df_filtered):,}** documents "
        f"({df_filtered['label'].value_counts().to_dict()})"
    )

    if not run_button:
        st.info("Configure settings in the sidebar, then press **Run topic analysis**.")
        return

    # ── Run model ─────────────────────────────────────────────────────────────
    texts = df_filtered["clean_text"].tolist()

    with st.spinner(f"Running {method}… this may take a minute."):
        if method == "LDA":
            result = run_lda(texts, n_topics=n_topics, n_top_words=n_top_words)
        else:
            result = run_bertopic(texts, n_topics=n_topics, n_top_words=n_top_words)

    if result.error:
        st.error(f"Topic modelling failed: {result.error}")
        return

    if not result.topics:
        st.warning("No topics were discovered. Try increasing the document count or reducing n_topics.")
        return

    st.success(f"Found **{len(result.topics)}** topics. {result.coherence_info}")

    # ── Section 1: Topic size bar chart ───────────────────────────────────────
    st.subheader("Topic sizes")
    size_df = pd.DataFrame(
        {
            "Topic": [t["label"][:40] for t in result.topics],
            "Documents": [t["size"] for t in result.topics],
        }
    ).set_index("Topic")
    st.bar_chart(size_df, use_container_width=True)

    # ── Section 2: Topic word cards ───────────────────────────────────────────
    st.subheader("Top words per topic")
    cols = st.columns(2)
    for i, topic in enumerate(result.topics):
        with cols[i % 2]:
            with st.expander(f"Topic {topic['id']}: {topic['label']}", expanded=(i < 4)):
                st.write("**Top words:** " + ", ".join(topic["top_words"]))
                if topic["docs"]:
                    st.markdown("**Sample documents:**")
                    for doc in topic["docs"][:3]:
                        snippet = doc[:200] + ("…" if len(doc) > 200 else "")
                        st.caption(snippet)

    # ── Section 3: Fake vs real distribution per topic ────────────────────────
    st.markdown("---")
    st.subheader("Fake vs real distribution per topic")

    if result.doc_assignments is not None:
        assignments = result.doc_assignments
        labels_col = df_filtered["label"].tolist()
        n_assign = min(len(assignments), len(labels_col))
        topic_ids_valid = {t["id"] for t in result.topics}

        rows = []
        for i in range(n_assign):
            tid = assignments[i]
            if tid in topic_ids_valid:
                rows.append({"topic_id": tid, "label": labels_col[i]})

        if rows:
            dist_df = pd.DataFrame(rows)
            pivot = (
                dist_df.groupby(["topic_id", "label"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            id_to_label = {t["id"]: t["label"] for t in result.topics}
            pivot["Topic"] = pivot["topic_id"].map(id_to_label)
            pivot = pivot.drop(columns=["topic_id"]).set_index("Topic")
            st.bar_chart(pivot, use_container_width=True)

    # ── Section 4: LDA heatmap ────────────────────────────────────────────────
    if method == "lda" and result.doc_topic_matrix is not None:
        st.markdown("---")
        st.subheader("Document–topic probability matrix")
        st.caption("LDA topic probabilities for a random sample of 60 documents. Darker = higher.")
        _doc_topic_heatmap(result, df_filtered)

    # ── Section 5: Infer topic for a custom article ───────────────────────────
    st.markdown("---")
    st.subheader("Infer topic for a custom article")
    st.caption("Paste a news title or snippet to find the closest trained topic.")

    user_text = st.text_input(
        "News title / snippet",
        placeholder="e.g. President signs new trade deal with China",
    )

    if user_text.strip():
        inferred = _infer_topic_lda(user_text, result)
        if inferred:
            st.success(
                f"Closest topic: **Topic {inferred['id']}** — {inferred['label']}\n\n"
                f"Top words: {', '.join(inferred['top_words'][:8])}"
            )
        else:
            st.info("No strong topic match found for this text.")
