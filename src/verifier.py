import os
import re
from functools import lru_cache
from typing import Dict, List

from src.config import (
    CLAIM_VERIFIER_DIR,
    NLI_DECISION_THRESHOLD,
    NLI_MAX_EVIDENCE,
    NLI_MAX_LENGTH,
    NLI_MODEL_NAME,
    NLI_NEI_MARGIN,
)
from src.runtime import configure_temp_dir


configure_temp_dir()

def _model_path() -> str:
    local_path = os.path.join(CLAIM_VERIFIER_DIR, "final")
    return local_path if os.path.isdir(local_path) else NLI_MODEL_NAME


def _load_nli_model_impl():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = _model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


try:
    import streamlit as st

    @st.cache_resource(show_spinner="Loading NLI verifier...")
    def load_nli_model():
        return _load_nli_model_impl()

except ImportError:

    @lru_cache(maxsize=1)
    def load_nli_model():
        return _load_nli_model_impl()


def _evidence_text(evidence: Dict) -> str:
    combined = evidence.get("combined_text", "")
    if combined and combined.strip():
        return combined.strip()
    title = str(evidence.get("title", "") or "")
    text = str(evidence.get("text", "") or evidence.get("description", "") or "")
    return f"{title}. {text}".strip(". ")


# ── Echo detection ────────────────────────────────────────────────────────
# When evidence text has very high word overlap with the claim, it is likely
# just another outlet echoing the same story rather than independently verifying
# it.  We discount entailment in that case.

_ECHO_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
    "that", "this", "those", "it", "its", "he", "she", "they",
    "his", "her", "their", "and", "or", "but", "not",
    "said", "says", "say", "has", "have", "had",
}


def _content_overlap(claim: str, evidence_text: str) -> float:
    """Fraction of claim content-words found in evidence (0..1).

    A high ratio (>0.7) suggests the evidence is just echoing the claim
    rather than providing independent verification.
    """
    claim_tokens = set(re.findall(r"\b\w+\b", claim.lower())) - _ECHO_STOPWORDS
    ev_tokens = set(re.findall(r"\b\w+\b", evidence_text.lower())) - _ECHO_STOPWORDS
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & ev_tokens) / len(claim_tokens)


def _label_index(id2label: Dict[int, str], label_name: str) -> int:
    label_name = label_name.lower()
    for idx, label in id2label.items():
        if label_name in str(label).lower():
            return int(idx)
    if len(id2label) == 3 and all(str(label).lower().startswith("label_") for label in id2label.values()):
        fallback = {"contrad": 0, "neutral": 1, "entail": 2}
        return fallback[label_name]
    raise ValueError(f"NLI model has no '{label_name}' label: {id2label}")


def _nli_scores(claim: str, evidence_list: List[Dict]) -> List[Dict]:
    import torch
    import re

    tokenizer, model, device = load_nli_model()
    
    # We will split each article into chunks so the NLI model isn't overwhelmed by long texts.
    article_chunks = []
    for evidence in evidence_list[:NLI_MAX_EVIDENCE]:
        text = _evidence_text(evidence)
        if not text:
            continue
            
        # Split into rough sentences/paragraphs
        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text) if len(c.strip()) > 20]
        # If no chunks (e.g. no punctuation), just use the whole text
        if not chunks:
            chunks = [text]
            
        # To provide a bit of context, we can group every 2 sentences together
        grouped_chunks = []
        for i in range(0, len(chunks), 2):
            grouped_chunks.append(" ".join(chunks[i:i+2]))
            
        for gc in grouped_chunks:
            article_chunks.append({
                "evidence": evidence,
                "text": gc
            })

    if not article_chunks:
        return []

    premises = [ac["text"] for ac in article_chunks]
    hypotheses = [claim] * len(article_chunks)
    
    inputs = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=NLI_MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).cpu()

    id2label = model.config.id2label
    entail_idx = _label_index(id2label, "entail")
    contra_idx = _label_index(id2label, "contrad")
    neutral_idx = _label_index(id2label, "neutral")

    # Group results back by article
    article_best_scores = {}
    
    for ac, prob in zip(article_chunks, probs):
        entail = float(prob[entail_idx])
        contra = float(prob[contra_idx])
        neutral = float(prob[neutral_idx])
        
        ev = ac["evidence"]
        ev_id = id(ev) # use object id as unique key
        
        signal = max(entail, contra)
        
        if ev_id not in article_best_scores or signal > article_best_scores[ev_id]["signal"]:
            credibility = ev.get("credibility_score", 0.4)
            overlap = _content_overlap(claim, ac["text"])
            
            article_best_scores[ev_id] = {
                "signal": signal,
                "score_dict": {
                    "evidence": ev,
                    "entailment": entail,
                    "contradiction": contra,
                    "neutral": neutral,
                    "confidence": max(entail, contra, neutral),
                    "overlap": round(overlap, 4),
                    "credibility": credibility,
                }
            }

    return [item["score_dict"] for item in article_best_scores.values()]


def _aggregate_scores(scores: List[Dict]) -> Dict:
    """Find the single piece of evidence that provides the strongest signal.
    
    Averaging all NLI probabilities is fundamentally flawed for fact-checking 
    because 1 perfectly matching article mixed with 4 irrelevant (Neutral) 
    articles will result in a 'Neutral' average. 
    
    Instead, we pick the evidence that has the highest credibility-weighted 
    entailment or contradiction score.
    """
    best_item = scores[0]
    max_signal = -1.0

    for item in scores:
        cred = max(item.get("credibility", 0.4), 0.1)
        # We only care about how strongly it supports or refutes the claim
        signal = max(item["entailment"], item["contradiction"]) * cred
        
        if signal > max_signal:
            max_signal = signal
            best_item = item

    return {
        "entailment": best_item["entailment"],
        "contradiction": best_item["contradiction"],
        "neutral": best_item["neutral"],
        "confidence": max(best_item["entailment"], best_item["contradiction"], best_item["neutral"]),
        "evidence": best_item["evidence"],
    }


def _verdict(score: Dict) -> tuple[str, float, str]:
    entail = score["entailment"]
    contra = score["contradiction"]
    neutral = score["neutral"]

    if entail >= NLI_DECISION_THRESHOLD and entail >= contra + NLI_NEI_MARGIN:
        return (
            "supported",
            entail,
            "NLI predicts that the retrieved evidence entails the claim.",
        )
    if contra >= NLI_DECISION_THRESHOLD and contra >= entail + NLI_NEI_MARGIN:
        return (
            "refuted",
            contra,
            "NLI predicts that the retrieved evidence contradicts the claim.",
        )
    return (
        "nei",
        max(neutral, min(entail, contra)),
        "NLI found related evidence, but not enough to clearly support or refute the claim.",
    )


def verify_claim(claim: str, evidence_list: List[Dict]) -> Dict:
    if not evidence_list:
        return {
            "label": "nei",
            "confidence": 0.2,
            "reason": "No retrieved evidence was available for this claim.",
            "best_evidence": None,
        }

    scores = _nli_scores(claim, evidence_list)
    if not scores:
        return {
            "label": "nei",
            "confidence": 0.2,
            "reason": "Retrieved evidence did not contain usable text for NLI.",
            "best_evidence": None,
        }

    agg = _aggregate_scores(scores)
    label, confidence, reason = _verdict(agg)
    return {
        "label": label,
        "confidence": round(min(confidence, 0.99), 4),
        "reason": reason,
        "best_evidence": agg["evidence"],
        "nli_scores": {
            "entailment": round(agg["entailment"], 4),
            "contradiction": round(agg["contradiction"], 4),
            "neutral": round(agg["neutral"], 4),
        },
    }
