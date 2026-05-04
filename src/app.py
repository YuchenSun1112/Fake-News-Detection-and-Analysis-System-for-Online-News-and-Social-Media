import streamlit as st

from cleaner import clean_article_text
from baseline_model import predict_baseline
from claim_extractor import extract_claims_progressively
from retriever import retrieve_evidence
from verifier import verify_claim
from aggregator import aggregate_results
from sentiment_analyser import analyse_evidence_sentiment
from topic_analysis_page import render_topic_analysis_page


st.set_page_config(
    page_title="Fact-based News Verification System",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Fact-based News Verification System")


# ─────────────────────────────────────────────────────────────────────────────
# Shared render helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_final_result(container, final_result, is_partial=False):
    with container.container():
        st.markdown("## Current Verdict" if is_partial else "## Final Verdict")
        verdict = final_result.get("article_verdict", "Unverified")
        verdict_conf = final_result.get("confidence", 0.0)
        verdict_reason = final_result.get("summary_reason", "No summary reason available.")

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

        # ── Sentiment analysis display ────────────────────────────────────────
        sentiment = final_result.get("sentiment")
        if sentiment and not is_partial:
            st.markdown("### Media Sentiment Analysis")
            label = sentiment.get("label", "neutral")
            pos = sentiment.get("positive", 0)
            neu = sentiment.get("neutral", 0)
            neg = sentiment.get("negative", 0)
            method = sentiment.get("method", "unknown")

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{pos:.1%}")
            col2.metric("Neutral", f"{neu:.1%}")
            col3.metric("Negative", f"{neg:.1%}")

            if label == "positive":
                st.success(f"Overall media tone: **Positive** — reporting around this topic tends to be supportive or confirmatory.")
            elif label == "negative":
                st.warning(f"Overall media tone: **Negative** — reporting around this topic tends to be critical or contradictory.")
            else:
                st.info(f"Overall media tone: **Neutral** — reporting around this topic is balanced or factual.")

            st.caption(f"Sentiment method: {method}")


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
            reason = verification.get("reason", "No strong supporting or refuting evidence found.")

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
                st.info(f"Verification: NOT ENOUGH INFORMATION | Confidence: {confidence:.2%}")

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


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_verify, tab_topic, tab_chat = st.tabs(["🔍 Article Verification", "📊 Topic Analysis", "💬 Chat"])

# ── Tab 1: Article Verification ───────────────────────────────────────────────
with tab_verify:
    st.markdown(
        "Paste a news article below. The system will extract claims, retrieve evidence, "
        "verify them, and generate a final verdict."
    )

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
                final_result = aggregate_results([], baseline_result)
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
                        aggregate_results(claim_results, baseline_result, run_sentiment=False),
                        is_partial=(idx < total_claims),
                    )

                    progress_value = 35 + int((idx / max(total_claims, 1)) * 65)
                    progress_bar.progress(min(progress_value, 100))

                final_result = aggregate_results(claim_results, baseline_result, run_sentiment=True)
                render_final_result(final_placeholder, final_result, is_partial=False)
                progress_bar.progress(100)
                status_placeholder.success("Analysis complete.")

# ── Tab 2: Topic Analysis ─────────────────────────────────────────────────────
with tab_topic:
    render_topic_analysis_page()

# ── Tab 3: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown(
        "Paste a news article and chat with the system to verify claims, "
        "check sentiment, or explore the content."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_article" not in st.session_state:
        st.session_state.chat_article = ""
    if "chat_result" not in st.session_state:
        st.session_state.chat_result = None

    with st.expander("📄 Paste article to analyse", expanded=(not st.session_state.chat_article)):
        chat_input_text = st.text_area(
            "Article text", height=180,
            placeholder="Paste the news article here...",
            key="chat_article_area",
        )
        col_a, col_b = st.columns(2)
        if col_a.button("Load article", type="primary"):
            if chat_input_text.strip():
                st.session_state.chat_article = chat_input_text.strip()
                st.session_state.chat_result = None
                st.session_state.chat_messages = [{
                    "role": "assistant",
                    "content": (
                        "Article loaded! You can ask me:\n"
                        "- **verify** — fact-check the claims\n"
                        "- **sentiment** — analyse media tone\n"
                        "- **claims** — list extracted claims\n"
                        "- **summary** — preview the article"
                    ),
                }]
                st.rerun()
        if col_b.button("Clear chat"):
            st.session_state.chat_article = ""
            st.session_state.chat_messages = []
            st.session_state.chat_result = None
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask about the article...", key="chat_user_input")

    if user_msg:
        if not st.session_state.chat_article:
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "Please paste an article first using the panel above.",
            })
            st.rerun()

        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        q = user_msg.lower()

        def _run_pipeline():
            if st.session_state.chat_result is not None:
                return st.session_state.chat_result
            cleaned = clean_article_text(st.session_state.chat_article)
            try:
                baseline = predict_baseline(cleaned)
            except Exception as exc:
                baseline = {"label": "unavailable", "confidence": 0.0,
                            "prob_fake": 0.0, "prob_real": 0.0, "error": str(exc)}
            claims = []
            for event in extract_claims_progressively(cleaned):
                claims = event.get("current_claims", claims)
            claim_results = []
            for claim in claims:
                retrieval = retrieve_evidence(claim["text"])
                verification = verify_claim(claim["text"], retrieval.get("results", []))
                claim_results.append({"claim": claim, "evidence": retrieval, "verification": verification})
            final = aggregate_results(claim_results, baseline, run_sentiment=True)
            result = {"final": final, "claims": claim_results, "baseline": baseline}
            st.session_state.chat_result = result
            return result

        if any(kw in q for kw in ["verify", "real", "fake", "true", "false", "check", "fact"]):
            with st.spinner("Running verification pipeline..."):
                r = _run_pipeline()
            final = r["final"]
            verdict = final["article_verdict"]
            conf = final["confidence"]
            reason = final["summary_reason"]
            stats = final["stats"]
            sentiment = final.get("sentiment")
            emoji = "✅" if verdict == "Likely True" else ("❌" if verdict == "Likely False" else "❓")
            lines = [
                f"{emoji} **Verdict: {verdict}** (Confidence: {conf:.1%})",
                "",
                reason,
                "",
                f"Supported: {stats['supported']} | Refuted: {stats['refuted']} | NEI: {stats['nei']}",
            ]
            if sentiment:
                s_label = sentiment['label'].capitalize()
                lines.append(f"\n**Media sentiment:** {s_label} — Positive {sentiment['positive']:.1%} / Neutral {sentiment['neutral']:.1%} / Negative {sentiment['negative']:.1%}")
            if r["claims"]:
                top = r["claims"][0]
                ev = top["verification"]
                lines.append(f"\n**Top claim:** _{top['claim']['text']}_")
                lines.append(f"→ {ev['label'].upper()} ({ev['confidence']:.1%}) — {ev['reason']}")
            reply = "\n".join(lines)

        elif any(kw in q for kw in ["sentiment", "tone", "emotion", "positive", "negative"]):
            with st.spinner("Analysing sentiment..."):
                if st.session_state.chat_result and st.session_state.chat_result["final"].get("sentiment"):
                    sentiment = st.session_state.chat_result["final"]["sentiment"]
                else:
                    from sentiment_analyser import analyse_sentiment
                    cleaned = clean_article_text(st.session_state.chat_article)
                    sentiment = analyse_sentiment([cleaned]).to_dict()
            label = sentiment["label"]
            emoji = "😊" if label == "positive" else ("😟" if label == "negative" else "😐")
            reply = (
                f"{emoji} **Sentiment: {label.capitalize()}**\n\n"
                f"| | Score |\n|---|---|\n"
                f"| Positive | {sentiment['positive']:.1%} |\n"
                f"| Neutral | {sentiment['neutral']:.1%} |\n"
                f"| Negative | {sentiment['negative']:.1%} |\n\n"
                f"The media coverage around this article has a predominantly **{label}** tone."
            )

        elif any(kw in q for kw in ["claim", "claims", "extract", "statement"]):
            with st.spinner("Extracting claims..."):
                cleaned = clean_article_text(st.session_state.chat_article)
                claims = []
                for event in extract_claims_progressively(cleaned):
                    claims = event.get("current_claims", claims)
            if claims:
                lines = [f"Found **{len(claims)}** verifiable claim(s):\n"]
                for i, c in enumerate(claims, 1):
                    lines.append(f"{i}. _{c['text']}_ (score: {c['score']:.1f})")
                lines.append("\nType **verify** to fact-check these claims.")
                reply = "\n".join(lines)
            else:
                reply = "No verifiable claims could be extracted from this article."

        elif any(kw in q for kw in ["summary", "summarise", "summarize", "about", "what is", "tell me"]):
            cleaned = clean_article_text(st.session_state.chat_article)
            preview = cleaned[:500] + ("..." if len(cleaned) > 500 else "")
            reply = f"Here's a preview of the loaded article:\n\n_{preview}_\n\nType **verify** or **sentiment** to analyse it."

        else:
            reply = (
                "I can help you with:\n"
                "- **verify** — fact-check the article's claims\n"
                "- **sentiment** — analyse the media tone\n"
                "- **claims** — list extracted claims\n"
                "- **summary** — preview the article\n\n"
                "What would you like to know?"
            )

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
