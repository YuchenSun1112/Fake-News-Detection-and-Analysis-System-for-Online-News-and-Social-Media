import os
import pandas as pd

from config import (
    GOSSIPCOP_FAKE,
    GOSSIPCOP_REAL,
    POLITIFACT_FAKE,
    POLITIFACT_REAL,
)
from src.cleaner import clean_article_text


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


def load_baseline_data() -> pd.DataFrame:
    files = [
        (GOSSIPCOP_REAL, 1),
        (POLITIFACT_REAL, 1),
        (GOSSIPCOP_FAKE, 0),
        (POLITIFACT_FAKE, 0),
    ]

    dfs = []
    for path, label in files:
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(normalize_news_columns(df, label))
        else:
            print(f"Warning: file not found -> {path}")

    if not dfs:
        raise FileNotFoundError("No dataset files were found in /data")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    return merged
