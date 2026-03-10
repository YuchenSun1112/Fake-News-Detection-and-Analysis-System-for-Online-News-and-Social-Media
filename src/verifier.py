import re
from typing import List, Dict


def _extract_numbers(text: str):
    return re.findall(r"\d+(?:\.\d+)?", text)


def _has_negation(text: str) -> bool:
    return bool(re.search(r"\b(no|not|never|false|denied|deny|refuted|fake|hoax)\b", text, flags=re.IGNORECASE))


def _token_overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", b.lower()))

    if not tokens_a:
        return 0.0

    overlap = tokens_a.intersection(tokens_b)
    return len(overlap) / max(len(tokens_a), 1)


def verify_claim(claim: str, evidence_list: List[Dict]) -> Dict:
    best_result = {
        "label": "not enough information",
        "confidence": 0.0,
        "best_evidence": None,
        "reason": "No strong supporting or refuting evidence found.",
    }

    claim_numbers = set(_extract_numbers(claim))
    claim_negation = _has_negation(claim)

    for evidence in evidence_list:
        ev_text = evidence["text"]
        overlap = _token_overlap_ratio(claim, ev_text)

        evidence_numbers = set(_extract_numbers(ev_text))
        evidence_negation = _has_negation(ev_text)

        # Simple support signal
        support_score = overlap
        if claim_numbers and claim_numbers.intersection(evidence_numbers):
            support_score += 0.2

        # Simple refute signal
        refute_score = 0.0
        if overlap > 0.25 and claim_negation != evidence_negation:
            refute_score += 0.45

        if claim_numbers and evidence_numbers and claim_numbers.isdisjoint(evidence_numbers) and overlap > 0.2:
            refute_score += 0.35

        if refute_score > support_score and refute_score > best_result["confidence"]:
            best_result = {
                "label": "refuted",
                "confidence": round(min(refute_score, 0.99), 4),
                "best_evidence": evidence,
                "reason": "The retrieved evidence overlaps semantically but conflicts in polarity or numeric details.",
            }

        elif support_score >= refute_score and support_score > best_result["confidence"] and support_score > 0.2:
            best_result = {
                "label": "supported",
                "confidence": round(min(support_score, 0.99), 4),
                "best_evidence": evidence,
                "reason": "The retrieved evidence shows substantial lexical overlap and consistent details.",
            }

    return best_result