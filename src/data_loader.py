import os
import pandas as pd

from config import (
    GOSSIPCOP_FAKE,
    GOSSIPCOP_REAL,
    POLITIFACT_FAKE,
    POLITIFACT_REAL,
)
from src.cleaner import clean_article_text

# ── LIAR dataset path (optional) ─────────────────────────────────────────────
# Download from: https://huggingface.co/datasets/liar
# Files needed: train.tsv, valid.tsv, test.tsv
# Place them in: data/liar/
_LIAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# LIAR label mapping:
# true, mostly-true, half-true → real (1)
# barely-true, false, pants-fire → fake (0)
_LIAR_REAL_LABELS = {"true", "mostly-true", "half-true"}
_LIAR_FAKE_LABELS = {"barely-true", "false", "pants-fire"}

# LIAR TSV columns (no header row)
_LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state_info", "party", "barely_true_count",
    "false_count", "half_true_count", "mostly_true_count",
    "pants_on_fire_count", "context",
]


def _pick_text_column(df: pd.DataFrame) -> str:
    candidates = ["text", "content", "article", "body", "title"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No valid text column found. Available columns: {list(df.columns)}")


def _pick_title_column(df: pd.DataFrame) -> str:
    if "title" in df.columns:
        return "title"
    return None


def normalize_news_columns(df: pd.DataFrame, label: int) -> pd.DataFrame:
    text_col = _pick_text_column(df)
    title_col = _pick_title_column(df)

    normalized = pd.DataFrame()
    normalized["title"] = df[title_col].fillna("") if title_col else ""
    normalized["text"] = df[text_col].fillna("").astype(str)
    normalized["label"] = label

    normalized["clean_text"] = normalized["text"].apply(clean_article_text)

    if title_col:
        normalized["final_text"] = (
            normalized["title"].astype(str).str.strip() + " [SEP] " + normalized["clean_text"]
        )
    else:
        normalized["final_text"] = normalized["clean_text"]

    normalized["final_text"] = normalized["final_text"].str.strip()
    return normalized


def _load_liar_split(tsv_path: str) -> pd.DataFrame:
    """Load one LIAR TSV split and return normalized DataFrame."""
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=_LIAR_COLUMNS,
        quoting=3,          # QUOTE_NONE — avoids quote-parsing issues
    )

    df["label_str"] = df["label"].str.strip().str.lower()
    df = df[df["label_str"].isin(_LIAR_REAL_LABELS | _LIAR_FAKE_LABELS)].copy()
    df["label_int"] = df["label_str"].apply(
        lambda x: 1 if x in _LIAR_REAL_LABELS else 0
    )

    # Use statement as text, context as title
    normalized = pd.DataFrame()
    normalized["title"] = df["context"].fillna("").astype(str)
    normalized["text"] = df["statement"].fillna("").astype(str)
    normalized["label"] = df["label_int"].values
    normalized["clean_text"] = normalized["text"].apply(clean_article_text)
    normalized["final_text"] = (
        normalized["title"].str.strip() + " [SEP] " + normalized["clean_text"]
    ).str.strip()

    return normalized


def load_liar_data() -> pd.DataFrame:
    """Load all available LIAR splits from data/liar/."""
    splits = ["train.tsv", "valid.tsv", "test.tsv"]
    dfs = []
    for split in splits:
        path = os.path.join(_LIAR_DIR, split)
        if os.path.exists(path):
            try:
                df = _load_liar_split(path)
                dfs.append(df)
                print(f"[data_loader] Loaded LIAR {split}: {len(df)} rows")
            except Exception as exc:
                print(f"[data_loader] Failed to load {split}: {exc}")
        else:
            print(f"[data_loader] LIAR split not found: {path}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_baseline_data() -> pd.DataFrame:
    """
    Load all available datasets:
    - GossipCop fake/real
    - PolitiFact fake/real
    - LIAR (train + valid + test) if available in data/liar/

    Returns a shuffled, balanced DataFrame with columns:
        final_text, label (0=fake, 1=real)
    """
    dfs = []

    # ── GossipCop + PolitiFact ────────────────────────────────────────────────
    gossip_politifact_files = [
        (GOSSIPCOP_REAL, 1),
        (POLITIFACT_REAL, 1),
        (GOSSIPCOP_FAKE, 0),
        (POLITIFACT_FAKE, 0),
    ]
    for path, label in gossip_politifact_files:
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(normalize_news_columns(df, label))
        else:
            print(f"[data_loader] Warning: file not found -> {path}")

    # ── LIAR ─────────────────────────────────────────────────────────────────
    liar_df = load_liar_data()
    if not liar_df.empty:
        dfs.append(liar_df)
        print(f"[data_loader] LIAR total: {len(liar_df)} rows")

    if not dfs:
        raise FileNotFoundError("No dataset files were found.")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["final_text"])
    merged = merged[merged["final_text"].str.strip().str.len() > 10]
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print class distribution
    counts = merged["label"].value_counts()
    print(f"[data_loader] Total: {len(merged)} rows | Real: {counts.get(1,0)} | Fake: {counts.get(0,0)}")

    return merged
