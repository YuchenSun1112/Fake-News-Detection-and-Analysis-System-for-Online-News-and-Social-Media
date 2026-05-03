"""
verifier.py — Claim verification using a zero-shot NLI model.

Model: cross-encoder/nli-deberta-v3-small  (fast, ~180 MB)
Fallback: lexical heuristics (original logic) if the model cannot load.

Label mapping
-------------
NLI entailment  → supported
NLI contradiction → refuted
NLI neutral     → not enough information
"""

import re
import os
from typing import List, Dict

# ── Optional NLI model ────────────────────────────────────────────────────────
_nli_pipeline = None
_NLI_MODEL = os.getenv("CLAIM_VERIFIER_MODEL", "cross-encoder/nli-deberta-v3-small")
_NLI_LOADED = False


def _get_nli_pipeline():
    global _nli_pipeline, _NLI_LOADED
    if _NLI_LOADED:
        return _nli_pipeline
    _NLI_LOADED = True
    try:
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model=_NLI_MODEL,
            device=-1,          # CPU; set to 0 for GPU
        )
        print(f"[verifier] NLI model loaded: {_NLI_MODEL}")
    except Exception as exc:
        print(f"[verifier] NLI model unavailable, using lexical fallback: {exc}")
        _nli_pipeline = None
    return _nli_pipeline


# ── Lexical helpers (fallback) ────────────────────────────────────────────────

def _extract_numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", str(text)))


def _has_negation(text: str) -> bool:
    return bool(re.search(
        r"\b(no|not|never|false|denied|deny|refuted|fake|hoax)\b",
        str(text), flags=re.IGNORECASE,
    ))


def _token_overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", b.lower()))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _lexical_verify(claim: str, evidence_list: List[Dict]) -> Dict:
    """Original heuristic verifier — used when NLI model is unavailable."""
    best = {
        "label": "not enough information",
        "confidence": 0.0,
        "best_evidence": None,
        "reason": "No strong supporting or refuting evidence found.",
    }

    claim_numbers = _extract_numbers(claim)
    claim_neg = _has_negation(claim)

    for ev in evidence_list:
        ev_text = ev.get("combined_text") or ev.get("text", "")
        overlap = _token_overlap_ratio(claim, ev_text)
        ev_numbers = _extract_numbers(ev_text)
        ev_neg = _has_negation(ev_text)

        support_score = overlap
        if claim_numbers and claim_numbers & ev_numbers:
            support_score += 0.2

        refute_score = 0.0
        if overlap > 0.25 and claim_neg != ev_neg:
            refute_score += 0.45
        if claim_numbers and ev_numbers and claim_numbers.isdisjoint(ev_numbers) and overlap > 0.2:
            refute_score += 0.35

        if refute_score > support_score and refute_score > best["confidence"]:
            best = {
                "label": "refuted",
                "confidence": round(min(refute_score, 0.99), 4),
                "best_evidence": ev,
                "reason": "Evidence overlaps semantically but conflicts in polarity or numeric details.",
            }
        elif support_score >= refute_score and support_score > best["confidence"] and support_score > 0.2:
            best = {
                "label": "supported",
                "confidence": round(min(support_score, 0.99), 4),
                "best_evidence": ev,
                "reason": "Evidence shows substantial lexical overlap and consistent details.",
            }

    return best


# ── NLI verifier ──────────────────────────────────────────────────────────────

_NLI_LABELS = ["supported", "refuted", "not enough information"]


def _nli_verify(claim: str, evidence_list: List[Dict], pipe) -> Dict:
    """
    For each evidence item, run zero-shot NLI with the claim as hypothesis
    and the evidence text as premise. Pick the evidence item that gives the
    highest non-NEI score.
    """
    best = {
        "label": "not enough information",
        "confidence": 0.0,
        "best_evidence": None,
        "reason": "No strong supporting or refuting evidence found.",
    }

    for ev in evidence_list:
        ev_text = (ev.get("combined_text") or ev.get("text", "")).strip()
        if not ev_text:
            continue

        premise = ev_text[:512]

        try:
            result = pipe(
                sequences=premise,
                candidate_labels=_NLI_LABELS,
                hypothesis_template="This text {} the following claim: " + claim,
            )
        except Exception:
            continue

        scores = dict(zip(result["labels"], result["scores"]))
        supported_score = scores.get("supported", 0.0)
        refuted_score = scores.get("refuted", 0.0)
        nei_score = scores.get("not enough information", 0.0)

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        decisive_score = max(supported_score, refuted_score)
        current_decisive = max(
            best["confidence"] if best["label"] != "not enough information" else 0.0,
            0.0,
        )

        if decisive_score > current_decisive and top_label != "not enough information":
            if top_label == "supported":
                reason = (
                    f"NLI model found supporting evidence "
                    f"(confidence {supported_score:.0%}). "
                    f"Source: {ev.get('source', 'Unknown')}."
                )
            else:
                reason = (
                    f"NLI model found contradicting evidence "
                    f"(confidence {refuted_score:.0%}). "
                    f"Source: {ev.get('source', 'Unknown')}."
                )

            best = {
                "label": top_label,
                "confidence": round(min(top_score, 0.99), 4),
                "best_evidence": ev,
                "reason": reason,
                "nli_scores": {
                    "supported": round(supported_score, 4),
                    "refuted": round(refuted_score, 4),
                    "nei": round(nei_score, 4),
                },
            }

    return best


# ── Public API ────────────────────────────────────────────────────────────────

def verify_claim(claim: str, evidence_list: List[Dict]) -> Dict:
    """
    Verify a single claim against a list of evidence dicts.

    Each evidence dict should contain at least one of:
      - "combined_text"  (preferred, title + description)
      - "text"

    Returns a dict with keys:
      label        : "supported" | "refuted" | "not enough information"
      confidence   : float 0–1
      best_evidence: the evidence dict that drove the verdict (or None)
      reason       : human-readable explanation
    """
    if not evidence_list:
        return {
            "label": "not enough information",
            "confidence": 0.0,
            "best_evidence": None,
            "reason": "No evidence was retrieved for this claim.",
        }

    pipe = _get_nli_pipeline()

    if pipe is not None:
        return _nli_verify(claim, evidence_list, pipe)
    else:
        return _lexical_verify(claim, evidence_list)
