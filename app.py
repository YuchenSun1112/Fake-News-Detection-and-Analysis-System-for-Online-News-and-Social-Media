import streamlit as st

from src.cleaner import clean_article_text
from src.baseline_model import predict_baseline
from src.claim_extractor import extract_claims
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

user_input = st.text_area(
    "News Content",
    height=300,
    placeholder="Paste the news article here...",
)


if st.button("🔍 Verify News"):
    if not user_input.strip():
        st.warning("Please enter some news content first.")
    else:
        with st.spinner("Analyzing article..."):
            cleaned_article = clean_article_text(user_input)

            # 1) Baseline classifier
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

            # 2) Extract claims
            claims = extract_claims(cleaned_article)

            # 3) Retrieve evidence + verify claims
            claim_results = []
            for claim in claims:
                retrieval_output = retrieve_evidence(claim["text"])

                search_query = retrieval_output.get("query", "")
                evidence_list = retrieval_output.get("results", [])
                retrieval_error = retrieval_output.get("error")

                verification = verify_claim(claim["text"], evidence_list)

                claim_results.append(
                    {
                        "claim": claim,
                        "search_query": search_query,
                        "retrieval_error": retrieval_error,
                        "evidences": evidence_list,
                        "verification": verification,
                    }
                )

            # 4) Aggregate
            final_result = aggregate_results(claim_results)

        # =========================
        # Display Final Result
        # =========================
        st.markdown("## Final Verdict")
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

        st.write(verdict_reason)

        stats = final_result.get("stats")
        if stats:
            st.write(
                f"Supported: {stats.get('supported', 0)} | "
                f"Refuted: {stats.get('refuted', 0)} | "
                f"NEI: {stats.get('nei', 0)}"
            )

        st.markdown("---")

        # =========================
        # Display Baseline Result
        # =========================
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

        st.markdown("---")

        # =========================
        # Display Claim Results
        # =========================
        st.markdown("## Extracted Claims")

        if not claims:
            st.warning("No verifiable claims were extracted.")
        else:
            for idx, item in enumerate(claim_results, start=1):
                claim = item["claim"]
                verification = item["verification"]
                search_query = item.get("search_query", "")
                retrieval_error = item.get("retrieval_error")

                st.markdown(f"### Claim {idx}")
                st.write(claim.get("text", ""))
                st.write(f"Claim score: {claim.get('score', 0):.2f}")

                if search_query:
                    st.write(f"Search query used: `{search_query}`")

                if retrieval_error:
                    st.error(f"Retriever error: {retrieval_error}")

                label = verification.get("label", "not enough information")
                confidence = verification.get("confidence", 0.0)
                reason = verification.get(
                    "reason",
                    "No strong supporting or refuting evidence found.",
                )

                if label == "supported":
                    st.success(
                        f"Verification: SUPPORTED | Confidence: {confidence:.2%}"
                    )
                elif label == "refuted":
                    st.error(
                        f"Verification: REFUTED | Confidence: {confidence:.2%}"
                    )
                else:
                    st.info(
                        f"Verification: NOT ENOUGH INFORMATION | "
                        f"Confidence: {confidence:.2%}"
                    )

                st.write(f"Reason: {reason}")

                # Best evidence
                best_ev = verification.get("best_evidence")
                if best_ev:
                    st.markdown("**Best Evidence**")
                    st.write(f"Source: {best_ev.get('source', 'Unknown')}")
                    if best_ev.get("published_at"):
                        st.write(f"Published: {best_ev.get('published_at')}")
                    if best_ev.get("title"):
                        st.write(f"Title: {best_ev.get('title')}")
                    if best_ev.get("url"):
                        st.write(f"URL: {best_ev.get('url')}")
                    if best_ev.get("text"):
                        st.write(best_ev.get("text", "")[:800])

                # All retrieved evidence
                with st.expander("Show retrieved evidence candidates"):
                    evidences = item.get("evidences", [])

                    if not evidences:
                        st.warning("No evidence returned from GNews.")
                    else:
                        for ev in evidences:
                            st.markdown(
                                f"""
**Source:** {ev.get('source', 'Unknown')}  
**Published:** {ev.get('published_at', '')}  
**Title:** {ev.get('title', '')}  
**URL:** {ev.get('url', '')}  
**API Rank Score:** {ev.get('api_rank_score', 0):.4f}  
**Local Match Score:** {ev.get('local_match_score', 0):.4f}  
**Final Score:** {ev.get('score', 0):.4f}

**Text:** {ev.get('text', '')[:500]}
"""
                            )

                st.markdown("---")