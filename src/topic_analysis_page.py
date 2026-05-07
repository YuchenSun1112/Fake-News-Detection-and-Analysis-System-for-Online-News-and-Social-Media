"""
topic_analysis_page.py — Streamlit page for news topic modelling.

Two sub-tabs:
  1. Analyse Article  — paste any article, get topic instantly (default)
  2. Dataset Explorer — explore topics in the training dataset

Single unified sidebar that switches content based on active tab.

Supports:
  - LDA  (always available via scikit-learn)
  - BERTopic (optional, requires bertopic + sentence-transformers)
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Optional BERTopic ─────────────────────────────────────────────────────────
_BERTOPIC_AVAILABLE = False
try:
    from bertopic import BERTopic
    _BERTOPIC_AVAILABLE = True
except ImportError:
    pass

# ── LDA ───────────────────────────────────────────────────────────────────────
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicResult:
    method: str
    n_topics: int
    topics: List[Dict]
    doc_topic_matrix: Optional[np.ndarray] = None
    doc_assignments: Optional[List[int]] = None
    coherence_info: str = ""
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS_EXTRA = {
    "say", "said", "says", "new", "year", "years", "day", "days",
    "just", "like", "make", "made", "want", "know", "time", "look",
    "going", "come", "think", "people", "man", "woman", "way",
    "via", "photo", "video", "news", "report", "update", "breaking",
    "read", "watch", "click", "source", "latest", "exclusive",
    # common English stopwords to filter from BERTopic
    "the", "and", "to", "of", "in", "is", "it", "was", "for",
    "on", "are", "as", "with", "his", "her", "they", "this",
    "that", "have", "had", "not", "but", "from", "or", "an",
    "be", "been", "has", "he", "she", "we", "you", "at", "by",
    "which", "will", "do", "did", "its", "all", "more", "about",
    "who", "so", "up", "out", "if", "than", "no", "my", "our",
    "can", "their", "what", "one", "also", "would", "when", "there",
}


def _clean_for_topic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_meaningful_word(word: str) -> bool:
    """Filter out stopwords and very short words from topic keywords."""
    return len(word) >= 3 and word not in _STOP_WORDS_EXTRA


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_dataset(max_per_class: int = 3000) -> pd.DataFrame:
    try:
        from src.config import GOSSIPCOP_FAKE, GOSSIPCOP_REAL, POLITIFACT_FAKE, POLITIFACT_REAL
    except ImportError:
        st.error("Cannot import config.py — make sure you run from the project root.")
        return pd.DataFrame()

    files = [
        (GOSSIPCOP_FAKE, "fake"), (GOSSIPCOP_REAL, "real"),
        (POLITIFACT_FAKE, "fake"), (POLITIFACT_REAL, "real"),
    ]
    dfs = []
    for path, label in files:
        try:
            df = pd.read_csv(path)
            text_col = next(
                (c for c in ["text", "content", "article", "body", "title"] if c in df.columns), None
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

def run_lda(texts: List[str], n_topics: int = 10, n_top_words: int = 10) -> TopicResult:
    stop_words_combined = list(
        CountVectorizer(stop_words="english").get_stop_words() | _STOP_WORDS_EXTRA
    )
    vectorizer = CountVectorizer(
        max_features=8000,
        min_df=max(2, len(texts) // 500),
        max_df=0.85,
        stop_words=stop_words_combined,
        ngram_range=(1, 2),
    )
    try:
        dtm = vectorizer.fit_transform(texts)
    except Exception as exc:
        return TopicResult(method="lda", n_topics=0, topics=[], error=str(exc))

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42,
        max_iter=15, learning_method="online", batch_size=256,
    )
    doc_topic_matrix = lda.fit_transform(dtm)
    doc_assignments = doc_topic_matrix.argmax(axis=1).tolist()
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_id, topic_vec in enumerate(lda.components_):
        top_indices = topic_vec.argsort()[: -n_top_words - 1 : -1]
        top_words = [feature_names[i] for i in top_indices]
        # Filter meaningful words for label
        meaningful = [w for w in top_words if _is_meaningful_word(w)]
        label = " · ".join(meaningful[:4]) if meaningful else " · ".join(top_words[:4])
        size = int((doc_topic_matrix.argmax(axis=1) == topic_id).sum())
        docs = [texts[i] for i, a in enumerate(doc_assignments) if a == topic_id][:5]
        topics.append({
            "id": topic_id, "label": label,
            "top_words": top_words, "size": size, "docs": docs,
        })

    topics = sorted(topics, key=lambda t: t["size"], reverse=True)
    for rank, t in enumerate(topics):
        t["rank"] = rank

    perplexity = lda.perplexity(dtm)
    return TopicResult(
        method="lda", n_topics=n_topics, topics=topics,
        doc_topic_matrix=doc_topic_matrix, doc_assignments=doc_assignments,
        coherence_info=f"LDA perplexity: {perplexity:.1f} (lower is better)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# BERTopic (with stopword filtering)
# ─────────────────────────────────────────────────────────────────────────────

def run_bertopic(
    texts: List[str], n_topics: int = 10,
    n_top_words: int = 10, embedding_model: str = "all-MiniLM-L6-v2",
) -> TopicResult:
    if not _BERTOPIC_AVAILABLE:
        return TopicResult(
            method="bertopic", n_topics=0, topics=[],
            error="BERTopic is not installed. Run: uv add bertopic sentence-transformers",
        )
    try:
        from sklearn.feature_extraction.text import CountVectorizer as CV
        # Pass stopword-aware vectorizer to BERTopic so it filters common words
        stop_words_combined = list(
            CV(stop_words="english").get_stop_words() | _STOP_WORDS_EXTRA
        )
        vectorizer_model = CV(
            stop_words=stop_words_combined,
            ngram_range=(1, 2),
            min_df=2,
        )
        model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=n_topics,
            top_n_words=n_top_words + 5,  # fetch extra, filter below
            vectorizer_model=vectorizer_model,
            verbose=False,
        )
        topic_ids, _ = model.fit_transform(texts)
        topic_info = model.get_topic_info()

        topics = []
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue
            top_words_tuples = model.get_topic(tid) or []
            # Filter stopwords from BERTopic output
            all_words = [w for w, _ in top_words_tuples]
            meaningful_words = [w for w in all_words if _is_meaningful_word(w)]
            top_words = meaningful_words[:n_top_words] if meaningful_words else all_words[:n_top_words]
            label = " · ".join(top_words[:4]) if top_words else f"Topic {tid}"
            size = int(row.get("Count", 0))
            docs = [texts[i] for i, t in enumerate(topic_ids) if t == tid][:5]
            topics.append({
                "id": tid, "label": label,
                "top_words": top_words, "size": size, "docs": docs, "rank": 0,
            })

        topics = sorted(topics, key=lambda t: t["size"], reverse=True)
        for rank, t in enumerate(topics):
            t["rank"] = rank

        doc_assignments = [int(t) if t != -1 else -1 for t in topic_ids]
        return TopicResult(
            method="bertopic", n_topics=len(topics), topics=topics,
            doc_assignments=doc_assignments,
            coherence_info=f"BERTopic — embedding model: {embedding_model}",
        )
    except Exception as exc:
        return TopicResult(method="bertopic", n_topics=0, topics=[], error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Shared inference + rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_topic(text: str, result: TopicResult) -> Optional[Dict]:
    clean = _clean_for_topic(text)
    tokens = set(clean.split())
    best_topic, best_score = None, -1.0
    for topic in result.topics:
        meaningful = [w for w in topic["top_words"] if _is_meaningful_word(w)]
        if not meaningful:
            continue
        overlap = len(tokens & set(meaningful))
        score = overlap / len(meaningful)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic if best_score > 0 else None


def _render_inference_result(
    result: TopicResult,
    df_filtered: Optional[pd.DataFrame],
    user_text: str,
) -> None:
    inferred = _infer_topic(user_text, result)
    if not inferred:
        st.info("No strong topic match found. Try using more text or increasing training documents.")
        return

    clean_input = _clean_for_topic(user_text)
    input_tokens = set(clean_input.split())
    meaningful_top_words = [w for w in inferred["top_words"] if _is_meaningful_word(w)]
    matched_words = sorted(input_tokens & set(meaningful_top_words))

    st.success(f"Closest topic: **Topic {inferred['id']}** — {inferred['label']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Topic ID", f"Topic {inferred['id']}")
    col2.metric("Matched keywords", len(matched_words))
    col3.metric("Topic size", f"{inferred['size']} docs")

    st.markdown("**Top words in this topic:**")
    st.write(", ".join(meaningful_top_words[:10]))

    if matched_words:
        st.markdown("**Keywords found in your article:**")
        st.write(", ".join(matched_words))

    if df_filtered is not None and result.doc_assignments is not None:
        labels_col = df_filtered["label"].tolist()
        topic_docs = [
            labels_col[i] for i, tid in enumerate(result.doc_assignments)
            if tid == inferred["id"] and i < len(labels_col)
        ]
        if topic_docs:
            fake_count = topic_docs.count("fake")
            real_count = topic_docs.count("real")
            total = len(topic_docs)
            fake_pct = fake_count / total if total else 0
            real_pct = real_count / total if total else 0

            st.markdown("**Fake/real distribution in this topic:**")
            col_f, col_r = st.columns(2)
            col_f.metric("Fake", f"{fake_count} ({fake_pct:.1%})")
            col_r.metric("Real", f"{real_count} ({real_pct:.1%})")

            if fake_pct > 0.65:
                st.warning(
                    f"⚠️ This topic has a high proportion of fake news ({fake_pct:.1%}) "
                    "in the training dataset. Articles on this topic may warrant extra scrutiny."
                )
            elif real_pct > 0.65:
                st.info(f"✅ This topic is predominantly real news ({real_pct:.1%}) in the training dataset.")
            else:
                st.info(f"Mixed fake/real distribution ({fake_pct:.1%} fake / {real_pct:.1%} real).")

    if inferred.get("docs"):
        with st.expander("Sample articles from this topic in the dataset"):
            for doc in inferred["docs"][:3]:
                st.caption(doc[:250] + ("…" if len(doc) > 250 else ""))
                st.markdown("---")


def _doc_topic_heatmap(result: TopicResult, df: pd.DataFrame) -> None:
    if result.doc_topic_matrix is None:
        return
    sample_size = min(60, len(df))
    sample_idx = np.random.choice(len(df), size=sample_size, replace=False)
    matrix_sample = result.doc_topic_matrix[sample_idx]
    heatmap_df = pd.DataFrame(
        matrix_sample,
        columns=[f"T{t['id']}" for t in result.topics],
        index=[f"doc{i}" for i in range(sample_size)],
    )
    st.dataframe(
        heatmap_df.style.background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True, height=300,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main page renderer — unified sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_topic_analysis_page() -> None:
    method_options = ["LDA"]
    if _BERTOPIC_AVAILABLE:
        method_options.append("BERTopic")

    # Track active sub-tab via session state
    if "topic_active_tab" not in st.session_state:
        st.session_state.topic_active_tab = "article"

    # ── Unified sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Topic analysis settings")
        method = st.selectbox("Method", method_options, key="topic_method")
        n_topics = st.slider("Number of topics", min_value=3, max_value=30, value=10, step=1, key="topic_n_topics")
        n_top_words = st.slider("Top words per topic", min_value=5, max_value=20, value=10, step=1, key="topic_n_top_words")

        # Show extra options depending on active tab
        if st.session_state.topic_active_tab == "article":
            max_docs = st.slider(
                "Training docs", min_value=200, max_value=3000, value=1000, step=200, key="topic_art_max_docs"
            )
            label_filter = ["fake", "real"]
        else:
            label_filter = st.multiselect(
                "Filter by label", options=["fake", "real"], default=["fake", "real"], key="topic_exp_label"
            )
            max_docs = st.slider(
                "Max documents", min_value=200, max_value=5000, value=2000, step=200, key="topic_exp_max_docs"
            )

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    sub_tab_article, sub_tab_explorer = st.tabs(["📰 Analyse Article", "📊 Dataset Explorer"])

    # ── Tab 1: Analyse Article ────────────────────────────────────────────────
    with sub_tab_article:
        st.session_state.topic_active_tab = "article"
        st.markdown("Paste any news article to identify its topic and see how it compares with the training dataset.")

        user_text = st.text_area(
            "News article",
            height=220,
            placeholder="Paste the full news article here...",
            key="art_article_input",
        )

        if st.button("Analyse topic", type="primary", key="art_analyse_btn"):
            if not user_text.strip():
                st.warning("Please paste an article first.")
            else:
                with st.spinner("Loading dataset…"):
                    df_full = _load_dataset()

                if df_full.empty:
                    st.error("No dataset files found. Make sure the CSV files exist in data/.")
                else:
                    df_sampled = df_full[df_full["label"].isin(label_filter)].copy()
                    df_sampled = df_sampled.sample(
                        n=min(max_docs, len(df_sampled)), random_state=42
                    ).reset_index(drop=True)
                    texts = df_sampled["clean_text"].tolist()

                    with st.spinner(f"Training {method} on {len(texts):,} documents…"):
                        if method == "LDA":
                            result = run_lda(texts, n_topics=n_topics, n_top_words=n_top_words)
                        else:
                            result = run_bertopic(texts, n_topics=n_topics, n_top_words=n_top_words)

                    if result.error:
                        st.error(f"Topic modelling failed: {result.error}")
                    elif not result.topics:
                        st.warning("No topics discovered. Try increasing training docs.")
                    else:
                        st.session_state["art_result"] = result
                        st.session_state["art_df"] = df_sampled
                        st.session_state["art_text"] = user_text

        if st.session_state.get("art_result") is not None:
            result = st.session_state["art_result"]
            df_sampled = st.session_state["art_df"]
            display_text = st.session_state.get("art_text", user_text)
            st.markdown("---")
            st.caption(f"Model: {result.coherence_info} | Trained on {len(df_sampled):,} documents")
            _render_inference_result(result, df_sampled, display_text)

    # ── Tab 2: Dataset Explorer ───────────────────────────────────────────────
    with sub_tab_explorer:
        st.session_state.topic_active_tab = "explorer"
        st.markdown("Explore the underlying topics in the GossipCop and PolitiFact training datasets.")

        run_button = st.button("Run topic analysis", type="primary", key="exp_run_btn")

        with st.spinner("Loading dataset…"):
            df_full = _load_dataset()

        if df_full.empty:
            st.error("No dataset files found.")
        else:
            df_filtered = df_full[df_full["label"].isin(label_filter)].copy()
            if len(df_filtered) > max_docs:
                df_filtered = df_filtered.sample(n=max_docs, random_state=42).reset_index(drop=True)

            st.caption(
                f"Dataset: **{len(df_filtered):,}** documents "
                f"({df_filtered['label'].value_counts().to_dict()})"
            )

            if "exp_result" not in st.session_state:
                st.session_state.exp_result = None

            if run_button:
                texts = df_filtered["clean_text"].tolist()
                with st.spinner(f"Running {method}… this may take a minute."):
                    if method == "LDA":
                        result = run_lda(texts, n_topics=n_topics, n_top_words=n_top_words)
                    else:
                        result = run_bertopic(texts, n_topics=n_topics, n_top_words=n_top_words)

                if result.error:
                    st.error(f"Topic modelling failed: {result.error}")
                elif not result.topics:
                    st.warning("No topics discovered.")
                else:
                    st.session_state.exp_result = result
                    st.session_state.exp_df = df_filtered

            if st.session_state.exp_result is None:
                st.info("Configure settings in the sidebar, then press **Run topic analysis**.")
            else:
                result = st.session_state.exp_result
                df_exp = st.session_state.get("exp_df", df_filtered)

                st.success(f"Found **{len(result.topics)}** topics. {result.coherence_info}")

                # Topic sizes
                st.subheader("Topic sizes")
                size_df = pd.DataFrame({
                    "Topic": [t["label"][:40] for t in result.topics],
                    "Documents": [t["size"] for t in result.topics],
                }).set_index("Topic")
                st.bar_chart(size_df, use_container_width=True)

                # Topic word cards
                st.subheader("Top words per topic")
                cols = st.columns(2)
                for i, topic in enumerate(result.topics):
                    with cols[i % 2]:
                        with st.expander(f"Topic {topic['id']}: {topic['label']}", expanded=(i < 4)):
                            meaningful = [w for w in topic["top_words"] if _is_meaningful_word(w)]
                            st.write("**Top words:** " + ", ".join(meaningful[:10]))
                            if topic["docs"]:
                                st.markdown("**Sample documents:**")
                                for doc in topic["docs"][:3]:
                                    st.caption(doc[:200] + ("…" if len(doc) > 200 else ""))

                # Fake vs real distribution
                st.markdown("---")
                st.subheader("Fake vs real distribution per topic")
                if result.doc_assignments is not None:
                    labels_col = df_exp["label"].tolist()
                    n_assign = min(len(result.doc_assignments), len(labels_col))
                    topic_ids_valid = {t["id"] for t in result.topics}
                    rows = [
                        {"topic_id": result.doc_assignments[i], "label": labels_col[i]}
                        for i in range(n_assign)
                        if result.doc_assignments[i] in topic_ids_valid
                    ]
                    if rows:
                        dist_df = pd.DataFrame(rows)
                        pivot = dist_df.groupby(["topic_id", "label"]).size().unstack(fill_value=0).reset_index()
                        id_to_label = {t["id"]: t["label"] for t in result.topics}
                        pivot["Topic"] = pivot["topic_id"].map(id_to_label)
                        pivot = pivot.drop(columns=["topic_id"]).set_index("Topic")
                        st.bar_chart(pivot, use_container_width=True)

                # LDA heatmap
                if method == "LDA" and result.doc_topic_matrix is not None:
                    st.markdown("---")
                    st.subheader("Document–topic probability matrix")
                    st.caption("LDA topic probabilities for a random sample of 60 documents.")
                    _doc_topic_heatmap(result, df_exp)

                # Article inference in explorer
                st.markdown("---")
                st.subheader("Analyse a custom article")
                st.caption("Paste an article to find its closest topic in the trained model.")
                exp_user_text = st.text_area(
                    "News article / snippet", height=180,
                    placeholder="Paste the news article here...",
                    key="exp_article_input",
                )
                if st.button("Analyse article topic", key="exp_article_btn"):
                    if not exp_user_text.strip():
                        st.warning("Please paste some text first.")
                    else:
                        _render_inference_result(result, df_exp, exp_user_text)
