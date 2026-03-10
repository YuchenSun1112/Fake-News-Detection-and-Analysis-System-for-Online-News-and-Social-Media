import re
from typing import List, Dict

from config import TOP_K_CLAIMS
from src.cleaner import clean_claim_text


def split_sentences(text: str) -> List[str]:
    text = str(text).strip()

    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r"(?<=[.!?])(?=\S)|(?<=[.!?])\s+", text)

    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _claim_score(sentence: str) -> float:
    score = 0.0

    # length
    if len(sentence.split()) >= 6:
        score += 1.0

    # numbers / dates
    if re.search(r"\d", sentence):
        score += 1.0

    # named-entity-like patterns
    if re.search(r"\b[A-Z][a-z]+\b", sentence):
        score += 0.8

    # reporting verbs
    if re.search(
        r"\b(said|announced|confirmed|reported|claimed|stated|according to|warned|showed|revealed)\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        score += 1.0

    # organizations / public institutions keywords
    if re.search(
        r"\b(government|who|cdc|president|ministry|united nations|tesla|google|apple|court|police|agency)\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        score += 1.0

    return score


def extract_claims(article_text: str, top_k: int = TOP_K_CLAIMS) -> List[Dict]:
    sentences = split_sentences(article_text)

    candidates = []
    for idx, sent in enumerate(sentences):
        clean_sent = clean_claim_text(sent)
        if len(clean_sent.split()) < 5:
            continue

        score = _claim_score(clean_sent)
        if score > 0:
            candidates.append(
                {
                    "claim_id": idx + 1,
                    "text": clean_sent,
                    "score": round(score, 4),
                }
            )

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]