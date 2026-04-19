import streamlit as st

from src.cleaner import clean_article_text
from src.baseline_model import predict_baseline
from src.claim_extractor import extract_claims_progressively
from src.retriever import retrieve_evidence
from src.verifier import verify_claim
from src.aggregator import aggregate_results


st.set_page_config(
    page_title="Fact-based News Verification System",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Fact-based News Verification System")
st.markdown(
    "Paste a news article below. The system will extract claims, retrieve evidence, "
    "verify them, and generate a final verdict."
)


def render_final_result(container, final_result, is_partial=False):
    with container.container():
        st.markdown("## Current Verdict" if is_partial else "## Final Verdict")

        verdict = final_result.get("article_verdict", "Unverified")
        verdict_conf = final_result.get("confidence", 0.0)
        verdict_reason = final_result.get(
            "summary_reason",
            "No summary reason available.",
        )

        if verdict == "Likely True":
            st.success(f"**{verdict}** | Confidence: **{verdict_conf:.2%}**")
        elif verdict == "Likely False":
            st.error(f"**{verdict}** | Confidence: **{verdict_conf:.2%}**")
        else:
            st.info(f"**{verdict}** | Confidence: **{verdict_conf:.2%}**")

        if is_partial:
            st.caption("Partial result based on the claims processed so far.")

        st.write(verdict_reason)

        stats = final_result.get("stats")
        if stats:
            st.write(
                f"Supported: {stats.get('supported', 0)} | "
                f"Refuted: {stats.get('refuted', 0)} | "
                f"NEI: {stats.get('nei', 0)}"
            )


def render_baseline_result(container, baseline_result):
    with container.container():
        st.markdown("## Baseline Classifier Result")

        if baseline_result["label"] != "unavailable":
            st.info(
                f"Baseline prediction: **{baseline_result['label'].upper()}** "
                f"(Confidence: {baseline_result['confidence']:.2%})"
            )
            st.write(
                f"Prob Fake: {baseline_result['prob_fake']:.2%} | "
                f"Prob Real: {baseline_result['prob_real']:.2%}"
            )
        else:
            st.warning(
                f"Baseline model unavailable: "
                f"{baseline_result.get('error', 'Unknown error')}"
            )


def render_extraction_progress(container, extraction_events, current_claims):
    with container.container():
        st.markdown("## Extraction Progress")

        if not extraction_events:
            st.info("Waiting for chunk summaries...")
        else:
            for event in extraction_events:
                chunk_id = event.get("chunk_id", "?")
                total_chunks = event.get("total_chunks", "?")
                chunk_text = event.get("chunk_text", "")
                chunk_claims = event.get("chunk_claims", [])

                with st.expander(f"Chunk {chunk_id}/{total_chunks}", expanded=(chunk_id == 1)):
                    preview = chunk_text[:400]
                    if len(chunk_text) > 400:
                        preview += "..."
                    st.write(preview)

                    if chunk_claims:
                        st.markdown("**Chunk summaries**")
                        for claim in chunk_claims:
                            st.write(f"- {claim.get('text', '')}")
                    else:
                        st.caption("No usable summary from this chunk.")

        st.markdown("### Current Claims")
        if current_claims:
            for claim in current_claims:
                st.write(
                    f"{claim.get('claim_id', '?')}. {claim.get('text', '')} "
                    f"(score: {claim.get('score', 0.0):.2f})"
                )
        else:
            st.caption("No claims selected yet.")


def render_claim_results(container, claim_results, total_claims):
    with container.container():
        st.markdown("## Claim Results")

        if not claim_results:
            st.info("Waiting for claim verification results...")
            return

        for idx, item in enumerate(claim_results, start=1):
            claim = item.get("claim", {})
            evidence_data = item.get("evidence", {})
            verification = item.get("verification", {})

            claim_text = claim.get("text", "")
            claim_score = claim.get("score", 0.0)
            search_query = evidence_data.get("query", "")
            retrieval_error = evidence_data.get("error")
            label = verification.get("label", "nei")
            confidence = verification.get("confidence", 0.0)
            reason = verification.get(
                "reason",
                "No strong supporting or refuting evidence found.",
            )

            st.markdown(f"### Claim {idx}/{total_claims}")
            st.write(claim_text)
            st.write(f"Claim score: {claim_score:.2f}")

            if search_query:
                st.write(f"Search queries used: `{search_query}`")

            if retrieval_error:
                st.error(f"Evidence retrieval error: {retrieval_error}")

            if label == "supported":
                st.success(f"Verification: SUPPORTED | Confidence: {confidence:.2%}")
            elif label == "refuted":
                st.error(f"Verification: REFUTED | Confidence: {confidence:.2%}")
            else:
                st.info(
                    f"Verification: NOT ENOUGH INFORMATION | "
                    f"Confidence: {confidence:.2%}"
                )

            st.write(f"Reason: {reason}")

            best_evidence = verification.get("best_evidence")
            if best_evidence:
                st.markdown("**Best Evidence**")
                st.write(f"Source: {best_evidence.get('source', 'Unknown')}")

                if best_evidence.get("published_at"):
                    st.write(f"Published: {best_evidence.get('published_at')}")

                if best_evidence.get("title"):
                    st.write(f"Title: {best_evidence.get('title')}")

                if best_evidence.get("url"):
                    st.markdown(f"[Open article]({best_evidence.get('url')})")

            st.markdown("---")

        remaining = max(total_claims - len(claim_results), 0)
        if remaining:
            st.caption(f"{remaining} claim(s) still processing...")

user_input = st.text_area(
    "News Content",
    height=300,
    placeholder="Paste the news article here...",
)


if st.button("Verify News"):
    if not user_input.strip():
        st.warning("Please enter some news content first.")
    else:
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        final_placeholder = st.empty()
        baseline_placeholder = st.empty()
        extraction_placeholder = st.empty()
        claims_placeholder = st.empty()

        status_placeholder.info("Cleaning article text...")
        progress_bar = progress_placeholder.progress(0)
        cleaned_article = clean_article_text(user_input)
        progress_bar.progress(5)

        status_placeholder.info("Running baseline classifier...")
        try:
            baseline_result = predict_baseline(cleaned_article)
        except Exception as e:
            baseline_result = {
                "label": "unavailable",
                "confidence": 0.0,
                "prob_fake": 0.0,
                "prob_real": 0.0,
                "error": str(e),
            }
        render_baseline_result(baseline_placeholder, baseline_result)
        progress_bar.progress(15)

        status_placeholder.info("Extracting retrieval-ready summaries...")
        extraction_events = []
        claims = []
        for event in extract_claims_progressively(cleaned_article):
            if event.get("stage") != "done":
                extraction_events.append(event)
            claims = event.get("current_claims", claims)
            render_extraction_progress(extraction_placeholder, extraction_events, claims)

        progress_bar.progress(35)

        if not claims:
            final_result = aggregate_results([])
            render_final_result(final_placeholder, final_result, is_partial=False)
            render_claim_results(claims_placeholder, [], 0)
            progress_bar.progress(100)
            status_placeholder.warning("No verifiable claims were extracted.")
        else:
            total_claims = len(claims)
            claim_results = []
            render_claim_results(claims_placeholder, claim_results, total_claims)

            for idx, claim in enumerate(claims, start=1):
                status_placeholder.info(
                    f"Processing claim {idx}/{total_claims}: retrieving evidence..."
                )

                retrieval_output = retrieve_evidence(claim["text"])
                verification = verify_claim(
                    claim["text"],
                    retrieval_output.get("results", []),
                )

                claim_results.append(
                    {
                        "claim": claim,
                        "evidence": retrieval_output,
                        "verification": verification,
                    }
                )

                render_claim_results(claims_placeholder, claim_results, total_claims)
                render_final_result(
                    final_placeholder,
                    aggregate_results(claim_results),
                    is_partial=(idx < total_claims),
                )

                progress_value = 35 + int((idx / max(total_claims, 1)) * 65)
                progress_bar.progress(min(progress_value, 100))

            final_result = aggregate_results(claim_results)
            render_final_result(final_placeholder, final_result, is_partial=False)
            progress_bar.progress(100)
            status_placeholder.success("Analysis complete.")
