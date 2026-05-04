"""
aggregator.py — Aggregate claim-level verification results into a final verdict.

Changes vs original
-------------------
1. Accepts an optional `baseline_result` dict (from predict_baseline).
2. Blends baseline prior (prob_fake / prob_real) with claim-level evidence.
3. The blend weight is configurable via the BASELINE_PRIOR_WEIGHT env var
   (default 0.25 — baseline contributes 25 %, claims 75 %).
4. Falls back gracefully when baseline is unavailable or unreliable.
"""

import os
from typing import List, Dict, Optional

from config import REFUTED_WEIGHT, SUPPORTED_WEIGHT, NEI_WEIGHT
from sentiment_analyser import analyse_evidence_sentiment, SentimentResult

# How much weight the baseline classifier's prediction gets.
# 0.0 = ignore baseline entirely; 1.0 = ignore claims entirely.
BASELINE_PRIOR_WEIGHT = float(os.getenv("BASELINE_PRIOR_WEIGHT", "0.25"))


def _verdict_from_scores(
    supported_score: float,
    refuted_score: float,
    supported_count: int,
    refuted_count: int,
    nei_count: int,
) -> tuple[str, float, str]:
    """Return (verdict, confidence, reason) from raw scores."""
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

    return verdict, confidence, reason


def aggregate_results(
    claim_results: List[Dict],
    baseline_result: Optional[Dict] = None,
    run_sentiment: bool = True,
) -> Dict:
    """
    Aggregate claim-level verifications into an article-level verdict.

    Parameters
    ----------
    claim_results : list of dicts, each with keys "claim", "evidence", "verification"
    baseline_result : optional dict from predict_baseline()
        Keys: label ("fake"|"real"|"unavailable"), prob_fake, prob_real, confidence

    Returns
    -------
    dict with keys:
        article_verdict  : "Likely True" | "Likely False" | "Unverified"
        confidence       : float 0–1
        summary_reason   : str
        stats            : {supported, refuted, nei, baseline_label}
        baseline_used    : bool
    """

    # ── No claims at all ────────────────────────────────────────────────────
    if not claim_results:
        # Fall back to baseline alone if available
        if baseline_result and baseline_result.get("label") not in (None, "unavailable"):
            bl = baseline_result
            verdict = "Likely False" if bl["label"] == "fake" else "Likely True"
            confidence = round(bl.get("confidence", 0.5), 4)
            return {
                "article_verdict": verdict,
                "confidence": confidence,
                "summary_reason": (
                    f"No verifiable claims were extracted. "
                    f"Baseline classifier predicts '{bl['label'].upper()}' "
                    f"(confidence {confidence:.0%})."
                ),
                "stats": {
                    "supported": 0,
                    "refuted": 0,
                    "nei": 0,
                    "baseline_label": bl["label"],
                },
                "baseline_used": True,
            }

        return {
            "article_verdict": "Unverified",
            "confidence": 0.0,
            "summary_reason": "No verifiable claims were extracted from the article.",
            "stats": {"supported": 0, "refuted": 0, "nei": 0, "baseline_label": None},
            "baseline_used": False,
        }

    # ── Accumulate claim scores ──────────────────────────────────────────────
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

    # ── Compute raw claim-based verdict ─────────────────────────────────────
    verdict, confidence, reason = _verdict_from_scores(
        supported_score, refuted_score,
        supported_count, refuted_count, nei_count,
    )

    # ── Blend in baseline prior ──────────────────────────────────────────────
    baseline_label = None
    baseline_used = False

    valid_baseline = (
        baseline_result is not None
        and baseline_result.get("label") not in (None, "unavailable")
        and BASELINE_PRIOR_WEIGHT > 0.0
    )

    if valid_baseline:
        bl = baseline_result
        baseline_label = bl["label"]
        bl_prob_fake = bl.get("prob_fake", 0.5)
        bl_prob_real = bl.get("prob_real", 0.5)
        bl_confidence = bl.get("confidence", 0.5)

        # Convert baseline to a [-1, +1] signal:
        #   +1 means "strongly real"  → pushes toward Likely True
        #   -1 means "strongly fake"  → pushes toward Likely False
        baseline_signal = bl_prob_real - bl_prob_fake  # range [-1, 1]

        # Convert claim verdict to the same scale
        if verdict == "Likely True":
            claim_signal = confidence
        elif verdict == "Likely False":
            claim_signal = -confidence
        else:
            claim_signal = 0.0

        # Weighted blend
        w_claim = 1.0 - BASELINE_PRIOR_WEIGHT
        w_base = BASELINE_PRIOR_WEIGHT
        blended_signal = w_claim * claim_signal + w_base * baseline_signal

        # Convert blended signal back to verdict
        THRESHOLD = 0.10  # minimum signal to make a call
        if blended_signal > THRESHOLD:
            verdict = "Likely True"
            confidence = min(abs(blended_signal), 0.99)
            reason += (
                f" Baseline classifier also predicted '{baseline_label.upper()}' "
                f"(confidence {bl_confidence:.0%}), reinforcing the verdict."
            )
        elif blended_signal < -THRESHOLD:
            verdict = "Likely False"
            confidence = min(abs(blended_signal), 0.99)
            reason += (
                f" Baseline classifier also predicted '{baseline_label.upper()}' "
                f"(confidence {bl_confidence:.0%}), reinforcing the verdict."
            )
        else:
            # Signals cancel out → Unverified
            verdict = "Unverified"
            confidence = 0.4
            reason = (
                "Claim-level evidence and baseline classifier produced conflicting signals; "
                "the article cannot be reliably classified."
            )

        baseline_used = True

    # ── Sentiment analysis on all retrieved evidence ────────────────────────────
    sentiment_result = None
    if run_sentiment:
        all_evidence = []
        for item in claim_results:
            ev_list = item.get("evidence", {}).get("results", [])
            all_evidence.extend(ev_list)
        if all_evidence:
            try:
                sentiment_result = analyse_evidence_sentiment(all_evidence)
            except Exception as exc:
                print(f"[aggregator] Sentiment analysis failed: {exc}")
                sentiment_result = None
        else:
            # 没有证据时，对 claim 文本本身做情感分析
            try:
                from sentiment_analyser import analyse_sentiment
                claim_texts = [
                    item["claim"]["text"]
                    for item in claim_results
                    if item.get("claim", {}).get("text")
                ]
                if claim_texts:
                    sentiment_result = analyse_sentiment(claim_texts)
            except Exception as exc:
                print(f"[aggregator] Fallback sentiment failed: {exc}")
                sentiment_result = None

    return {
        "article_verdict": verdict,
        "confidence": round(min(confidence, 0.99), 4),
        "summary_reason": reason,
        "stats": {
            "supported": supported_count,
            "refuted": refuted_count,
            "nei": nei_count,
            "baseline_label": baseline_label,
        },
        "baseline_used": baseline_used,
        "sentiment": sentiment_result.to_dict() if sentiment_result else None,
    }
