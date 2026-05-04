"""
sentiment_analyser.py — Sentiment analysis on news evidence text.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest  (~480 MB)
Labels: positive / neutral / negative
Fallback: VADER lexical scorer (no model download needed)

Public API
----------
analyse_sentiment(texts: list[str]) -> SentimentResult
analyse_evidence_sentiment(evidence_list: list[dict]) -> SentimentResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    label: str                        # "positive" | "neutral" | "negative"
    positive: float = 0.0
    neutral: float = 0.0
    negative: float = 0.0
    method: str = "unknown"           # "transformer" | "vader" | "lexical"
    per_text: List[Dict] = field(default_factory=list)

    @property
    def dominant_score(self) -> float:
        return max(self.positive, self.neutral, self.negative)

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "positive": round(self.positive, 4),
            "neutral": round(self.neutral, 4),
            "negative": round(self.negative, 4),
            "dominant_score": round(self.dominant_score, 4),
            "method": self.method,
            "per_text": self.per_text,
        }


# ── Transformer sentiment pipeline ───────────────────────────────────────────

_sentiment_pipeline = None
_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_SENTIMENT_LOADED = False


def _get_sentiment_pipeline():
    global _sentiment_pipeline, _SENTIMENT_LOADED
    if _SENTIMENT_LOADED:
        return _sentiment_pipeline
    _SENTIMENT_LOADED = True
    try:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=_SENTIMENT_MODEL,
            device=-1,
            truncation=True,
            max_length=512,
        )
        print(f"[sentiment] Model loaded: {_SENTIMENT_MODEL}")
    except Exception as exc:
        print(f"[sentiment] Transformer model unavailable, using fallback: {exc}")
        _sentiment_pipeline = None
    return _sentiment_pipeline


# ── VADER fallback ────────────────────────────────────────────────────────────

def _vader_scores(text: str) -> Dict:
    """Return VADER compound + mapped label. Installs nltk data lazily."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()

        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        pos = (compound + 1) / 2        # map [-1,1] → [0,1]
        neg = 1.0 - pos
        neu = 1.0 - abs(compound)
        total = pos + neg + neu
        return {
            "label": label,
            "positive": pos / total,
            "neutral": neu / total,
            "negative": neg / total,
        }
    except Exception:
        return {"label": "neutral", "positive": 0.33, "neutral": 0.34, "negative": 0.33}


# ── Simple lexical fallback (no dependencies) ─────────────────────────────────

_POS_WORDS = {
    "confirmed", "verified", "true", "accurate", "correct", "supports",
    "proves", "evidence", "fact", "real", "legitimate", "genuine",
    "approved", "success", "growth", "positive", "agree", "support",
}
_NEG_WORDS = {
    "false", "fake", "denied", "refuted", "wrong", "incorrect", "misleading",
    "disproven", "hoax", "fraud", "fabricated", "debunked", "lie", "lying",
    "disputed", "contradicts", "claims", "alleged", "unverified", "suspect",
}


def _lexical_scores(text: str) -> Dict:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    pos_hits = len(tokens & _POS_WORDS)
    neg_hits = len(tokens & _NEG_WORDS)
    total = pos_hits + neg_hits + 1  # +1 avoid div-zero, biases toward neutral
    pos = pos_hits / total
    neg = neg_hits / total
    neu = 1.0 - pos - neg
    if pos > neg and pos > 0.3:
        label = "positive"
    elif neg > pos and neg > 0.3:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "positive": pos, "neutral": neu, "negative": neg}


# ── Core analysis functions ───────────────────────────────────────────────────

def _score_one(text: str, pipe) -> Dict:
    """Score a single text snippet, return label + scores dict."""
    snippet = text[:512].strip()
    if not snippet:
        return {"label": "neutral", "positive": 0.33, "neutral": 0.34, "negative": 0.33}

    if pipe is not None:
        try:
            result = pipe(snippet)[0]
            raw_label = result["label"].lower()
            # Model uses: positive / neutral / negative (or label_0/1/2)
            label_map = {
                "label_0": "negative", "label_1": "neutral", "label_2": "positive",
                "positive": "positive", "neutral": "neutral", "negative": "negative",
            }
            label = label_map.get(raw_label, "neutral")
            score = result["score"]
            pos = score if label == "positive" else (1 - score) / 2
            neg = score if label == "negative" else (1 - score) / 2
            neu = 1.0 - pos - neg
            return {"label": label, "positive": max(pos, 0), "neutral": max(neu, 0), "negative": max(neg, 0)}
        except Exception:
            pass

    # Try VADER, then lexical
    try:
        return _vader_scores(snippet)
    except Exception:
        return _lexical_scores(snippet)


def analyse_sentiment(texts: List[str]) -> SentimentResult:
    """
    Analyse sentiment across a list of text snippets.
    Returns an aggregated SentimentResult.
    """
    if not texts:
        return SentimentResult(label="neutral", positive=0.33, neutral=0.34, negative=0.33, method="none")

    pipe = _get_sentiment_pipeline()
    method = "transformer" if pipe is not None else "vader"

    per_text = []
    for text in texts:
        scored = _score_one(text, pipe)
        per_text.append(scored)

    # Average scores across all texts
    n = len(per_text)
    avg_pos = sum(t["positive"] for t in per_text) / n
    avg_neu = sum(t["neutral"] for t in per_text) / n
    avg_neg = sum(t["negative"] for t in per_text) / n

    if avg_pos >= avg_neg and avg_pos >= avg_neu:
        label = "positive"
    elif avg_neg >= avg_pos and avg_neg >= avg_neu:
        label = "negative"
    else:
        label = "neutral"

    return SentimentResult(
        label=label,
        positive=round(avg_pos, 4),
        neutral=round(avg_neu, 4),
        negative=round(avg_neg, 4),
        method=method,
        per_text=per_text,
    )


def analyse_evidence_sentiment(evidence_list: List[Dict]) -> SentimentResult:
    """
    Convenience wrapper: extract text from evidence dicts and analyse sentiment.
    Uses combined_text if available, otherwise falls back to title + text.
    """
    texts = []
    for ev in evidence_list:
        text = ev.get("combined_text") or f"{ev.get('title', '')} {ev.get('text', '')}"
        if text.strip():
            texts.append(text.strip())

    if not texts:
        return SentimentResult(label="neutral", positive=0.33, neutral=0.34, negative=0.33, method="none")

    return analyse_sentiment(texts)
