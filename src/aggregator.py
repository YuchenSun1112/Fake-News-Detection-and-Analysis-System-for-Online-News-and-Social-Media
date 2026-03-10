from typing import List, Dict

from config import REFUTED_WEIGHT, SUPPORTED_WEIGHT, NEI_WEIGHT


def aggregate_results(claim_results: List[Dict]) -> Dict:
    if not claim_results:
        return {
            "article_verdict": "Unverified",
            "confidence": 0.0,
            "summary_reason": "No verifiable claims were extracted from the article.",
        }

    supported_score = 0.0
    refuted_score = 0.0
    nei_score = 0.0

    supported_count = 0
    refuted_count = 0
    nei_count = 0

    for item in claim_results:
        label = item["verification"]["label"]
        confidence = item["verification"]["confidence"]

        if label == "supported":
            supported_score += confidence * SUPPORTED_WEIGHT
            supported_count += 1
        elif label == "refuted":
            refuted_score += confidence * REFUTED_WEIGHT
            refuted_count += 1
        else:
            nei_score += max(confidence, 0.2) * NEI_WEIGHT
            nei_count += 1

    if refuted_score > supported_score and refuted_count >= 1:
        verdict = "Likely False"
        confidence = refuted_score / max(refuted_count, 1)
        reason = f"{refuted_count} key claim(s) were refuted by retrieved evidence."
    elif supported_score >= refuted_score and supported_count >= 1:
        verdict = "Likely True"
        confidence = supported_score / max(supported_count, 1)
        reason = f"{supported_count} key claim(s) were supported by retrieved evidence."
    else:
        verdict = "Unverified"
        confidence = 0.4
        reason = "Available evidence is insufficient to verify the main claims."

    return {
        "article_verdict": verdict,
        "confidence": round(min(confidence, 0.99), 4),
        "summary_reason": reason,
        "stats": {
            "supported": supported_count,
            "refuted": refuted_count,
            "nei": nei_count,
        },
    }