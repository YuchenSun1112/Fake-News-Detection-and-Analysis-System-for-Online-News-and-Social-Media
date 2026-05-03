import os
import re
from typing import List, Dict, Optional

from config import TOP_K_CLAIMS
from src.cleaner import clean_claim_text

try:
    from transformers import pipeline
except Exception:
    pipeline = None


# =========================================================
# Configuration
# =========================================================
CLAIM_EXTRACTOR_MODE = os.getenv("CLAIM_EXTRACTOR_MODE", "model").strip().lower()
CLAIM_EXTRACTOR_MODEL = os.getenv("CLAIM_EXTRACTOR_MODEL", "google/flan-t5-base").strip()

MIN_SENT_WORDS = int(os.getenv("MIN_SENT_WORDS", "5"))
MIN_CLAIM_WORDS = int(os.getenv("MIN_CLAIM_WORDS", "4"))
MAX_CLAIM_WORDS = int(os.getenv("MAX_CLAIM_WORDS", "36"))
MAX_INPUT_CHARS_PER_SENT = int(os.getenv("MAX_INPUT_CHARS_PER_SENT", "700"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "96"))
MAX_INPUT_CHARS_PER_SUMMARY_CHUNK = int(os.getenv("MAX_INPUT_CHARS_PER_SUMMARY_CHUNK", "900"))
MAX_SUMMARY_CHUNKS = int(os.getenv("MAX_SUMMARY_CHUNKS", "4"))
MIN_SUMMARY_CHUNK_CHARS = int(os.getenv("MIN_SUMMARY_CHUNK_CHARS", "220"))
MAX_RETRIEVAL_SUMMARIES_PER_CHUNK = int(os.getenv("MAX_RETRIEVAL_SUMMARIES_PER_CHUNK", "3"))

# Maximum number of atomic claims generated from one sentence
MAX_CLAIMS_PER_SENTENCE = int(os.getenv("MAX_CLAIMS_PER_SENTENCE", "4"))

# Similarity threshold used for deduplication
DEDUP_SIM_THRESHOLD = float(os.getenv("CLAIM_DEDUP_SIM_THRESHOLD", "0.82"))

_SEQ2SEQ_EXTRACTOR = None
_SEQ2SEQ_LOAD_FAILED = False


# =========================================================
# Sentence splitting
# =========================================================
def split_sentences(text: str) -> List[str]:
    """
    Split article text into sentences with a lightweight regex rule.
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])(?=\S)|(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _split_text_by_words(text: str, max_chars: int) -> List[str]:
    """
    Split text into approximate char-limited chunks while preserving word boundaries.
    """
    words = _normalize_spaces(text).split()
    if not words:
        return []

    chunks = []
    current = []
    current_len = 0

    for word in words:
        extra_len = len(word) + (1 if current else 0)
        if current and current_len + extra_len > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra_len

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _chunk_article_for_summary(article_text: str) -> List[str]:
    """
    Group the article into retrieval-friendly chunks for document-level seq2seq summarization.
    """
    article_text = _normalize_spaces(article_text)
    if not article_text:
        return []

    sentences = split_sentences(article_text)
    if not sentences:
        return _split_text_by_words(article_text, MAX_INPUT_CHARS_PER_SUMMARY_CHUNK)[:MAX_SUMMARY_CHUNKS]

    chunks = []
    current_sentences = []
    current_len = 0

    for sentence in sentences:
        sentence = _normalize_spaces(sentence)
        if not sentence:
            continue

        if len(sentence) > MAX_INPUT_CHARS_PER_SUMMARY_CHUNK:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_len = 0

            chunks.extend(_split_text_by_words(sentence, MAX_INPUT_CHARS_PER_SUMMARY_CHUNK))
            if len(chunks) >= MAX_SUMMARY_CHUNKS:
                return chunks[:MAX_SUMMARY_CHUNKS]
            continue

        extra_len = len(sentence) + (1 if current_sentences else 0)
        if current_sentences and current_len + extra_len > MAX_INPUT_CHARS_PER_SUMMARY_CHUNK:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
            current_len = len(sentence)
        else:
            current_sentences.append(sentence)
            current_len += extra_len

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    if len(chunks) >= 2 and len(chunks[-1]) < MIN_SUMMARY_CHUNK_CHARS:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks[:MAX_SUMMARY_CHUNKS]


# =========================================================
# Basic utilities
# =========================================================
def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _normalize_for_dedup(text: str) -> str:
    """
    Normalize text for duplicate detection.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[\"'“”‘’]", "", text)
    text = re.sub(r"[^a-z0-9\s%.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _jaccard_similarity(a: str, b: str) -> float:
    """
    Compute token-level Jaccard similarity.
    """
    sa = set(_normalize_for_dedup(a).split())
    sb = set(_normalize_for_dedup(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?%?", str(text))


def _extract_capitalized_entities(text: str) -> List[str]:
    """
    Extract simple capitalized entity-like spans.
    """
    blocked_singletons = {
        "A", "An", "The", "It", "This", "That", "These", "Those",
        "He", "She", "They", "We", "You", "I",
        "His", "Her", "Their", "Its",
        "Previously", "Currently", "Formerly", "Meanwhile", "However", "Later",
    }
    spans = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", str(text))
    seen = set()
    results = []
    for span in spans:
        if span in blocked_singletons:
            continue
        key = span.lower()
        if key not in seen:
            seen.add(key)
            results.append(span)
    return results


def _pick_context_entity(text: str) -> Optional[str]:
    """
    Pick a likely entity that can replace vague follow-up subjects such as "the airline".
    """
    entities = _extract_capitalized_entities(text)
    if not entities:
        return None

    preferred_patterns = [
        r"\b(airlines?|airways|administration|department|agency|company|court|ministry)\b",
    ]
    for entity in entities:
        if any(re.search(pattern, entity, flags=re.IGNORECASE) for pattern in preferred_patterns):
            return entity

    return entities[0]


def _replace_vague_subject_with_entity(text: str, entity: Optional[str]) -> str:
    if not entity:
        return text

    replacements = [
        r"^(the airline|airline)\b",
        r"^(the company|company)\b",
        r"^(the agency|agency)\b",
        r"^(the department|department)\b",
    ]
    updated = text
    for pattern in replacements:
        updated = re.sub(pattern, entity, updated, flags=re.IGNORECASE)
    return updated


def _prefer_context_entity(current: Optional[str], candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return current
    if not current:
        return candidate

    current_terms = len(current.split())
    candidate_terms = len(candidate.split())
    if candidate_terms >= current_terms:
        return candidate

    return current


def _has_strong_factual_signal(text: str) -> bool:
    """
    Detect whether a sentence/claim looks fact-like enough to verify.
    """
    patterns = [
        r"\d",
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
        r"\b(said|announced|confirmed|reported|claimed|stated|revealed|showed|warned|"
        r"arrested|charged|killed|injured|approved|banned|won|lost|died|cut|cuts|"
        r"increased|decreased|launched|signed|closed|opened|issued|beat|defeated|"
        r"elected|fined|suspended|sentenced)\b",
        r"\b(government|president|ministry|court|police|agency|who|cdc|fda|nasa|"
        r"united nations|tesla|google|apple|meta|amazon|microsoft)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _is_question_or_low_value_sentence(text: str) -> bool:
    """
    Filter out obvious non-claim sentences.
    """
    s = _normalize_spaces(text)

    if not s:
        return True

    if s.endswith("?"):
        return True

    if re.search(
        r"\b(i think|in my opinion|maybe|perhaps|possibly|could be|might be|seems like|"
        r"it is possible|many believe|some believe|people believe|rumou?r has it|"
        r"rumou?rs? say)\b",
        s,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def _remove_attribution_prefix(text: str) -> str:
    """
    Remove weak attribution/opening phrases while preserving factual core.
    """
    patterns = [
        r"^according to [^,]+,\s*",
        r"^reportedly,\s*",
        r"^it said that\s*",
        r"^sources said\s*",
        r"^officials said\s*",
        r"^posts online claimed that\s*",
        r"^social media posts claimed that\s*",
        r"^witnesses said\s*",
        r"^[A-Z][A-Za-z.\- ]+,\s*[A-Z][a-z]+\s+\d{1,2}\s*:\s*",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _strip_leading_discourse_markers(text: str) -> str:
    """
    Remove stylistic sentence openers that are not useful for retrieval.
    """
    return re.sub(
        r"^(ultimately|meanwhile|however|moreover|furthermore|therefore|"
        r"nonetheless|nevertheless|in the end|at the same time)\s*,?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _contains_subjective_language(text: str) -> bool:
    """
    Detect subjective, evaluative, or narrative wording that hurts verification quality.
    """
    return bool(
        re.search(
            r"\b(fine strike|huge blow|massive blow|stunning|dramatic|remarkable|"
            r"steamrollered|status as|powerhouse|disappoint(ed|ing)? the crowd|"
            r"looked for an equaliser|adaptability and character|on the back foot|"
            r"vast majority of the crowd)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _contains_background_language(text: str) -> bool:
    """
    Detect broad narrative/background wording that is usually less useful than concrete events.
    """
    return bool(
        re.search(
            r"\b("
            r"tumultuous ride|pioneer of|has endured|have endured|became the pioneer|"
            r"history of|long-running|long running|over the years|for decades|"
            r"legacy|reputation|status as"
            r")\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_finite_verb(text: str) -> bool:
    """
    Require at least one finite verb or auxiliary so we keep complete factual statements.
    """
    return bool(
        re.search(
            r"\b("
            r"is|are|was|were|be|been|being|"
            r"has|have|had|"
            r"do|does|did|"
            r"will|would|can|could|may|might|must|should|"
            r"says|said|announces|announced|confirms|confirmed|reports|reported|"
            r"claims|claimed|states|stated|reveals|revealed|shows|showed|"
            r"warns|warned|approves|approved|bans|banned|wins|won|loses|lost|"
            r"kills|killed|injures|injured|cuts|cut|raises|raised|falls|fell|"
            r"launches|launched|signs|signed|opens|opened|closes|closed|"
            r"issues|issued|elects|elected|fines|fined|suspends|suspended|"
            r"sentences|sentenced|creates|created|causes|caused|"
            r"files|filed|blocks|blocked|blames|blamed|cancels|canceled|cancelled|"
            r"merges|merged|fails|failed"
            r")\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_specific_search_anchor(text: str) -> bool:
    """
    Keep claims that mention a concrete entity, number, institution, or country-like anchor.
    """
    if _extract_numbers(text):
        return True

    if _extract_capitalized_entities(text):
        return True

    return bool(
        re.search(
            r"\b("
            r"government|president|prime minister|ministry|court|police|agency|"
            r"army|military|parliament|congress|senate|eu|un|nato|who|cdc|fda|nasa|"
            r"india|china|maldives|russia|ukraine|israel|iran|gaza|taiwan|"
            r"japan|france|germany|britain|united states|u\.s\."
            r")\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _starts_with_vague_pronoun(text: str) -> bool:
    return bool(re.match(r"^(it|this|that|these|those|they|he|she)\b", text, flags=re.IGNORECASE))


def _looks_like_fragment(text: str) -> bool:
    """
    Reject phrases such as 'Previously critical of ...' that are not standalone claims.
    """
    text = _normalize_spaces(text)
    if not text:
        return True

    if not _has_finite_verb(text):
        return True

    return bool(
        re.match(
            r"^(previously|currently|formerly|reportedly|allegedly|meanwhile|however|"
            r"later|earlier)\b",
            text,
            flags=re.IGNORECASE,
        ) and not _has_specific_search_anchor(text)
    )


def _clean_claim_candidate(text: str) -> str:
    """
    Normalize a candidate claim into a concise factual style.
    """
    text = clean_claim_text(text)
    text = _normalize_spaces(text)
    text = _remove_attribution_prefix(text)
    text = _strip_leading_discourse_markers(text)

    text = re.sub(r"^[,;:\-]+", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    # Remove trailing attribution fragments if they remain
    text = re.sub(
        r",?\s*(according to [^,]+|sources said|officials said)$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    return text


# =========================================================
# Rule-based fallback extraction
# =========================================================
def _rule_claim_score(text: str) -> float:
    """
    Assign a heuristic score to a candidate claim.
    """
    score = 0.0
    tokens = text.split()

    if len(tokens) >= 5:
        score += 1.0
    if 6 <= len(tokens) <= 18:
        score += 0.8
    elif len(tokens) <= 28:
        score += 0.4

    if re.search(r"\d", text):
        score += 1.0
    if re.search(r"\b\d+(?:\.\d+)?%\b", text):
        score += 0.4

    if _extract_capitalized_entities(text):
        score += 0.8

    if re.search(
        r"\b(announced|confirmed|reported|revealed|approved|banned|won|lost|beat|"
        r"defeated|arrested|charged|killed|injured|signed|cut|raised|fell|elected|"
        r"suspended|sentenced|launched|issued)\b",
        text,
        flags=re.IGNORECASE,
    ):
        score += 1.0

    if _has_strong_factual_signal(text):
        score += 0.8

    if not _contains_subjective_language(text):
        score += 0.4

    if _contains_background_language(text):
        score -= 1.2

    # Penalize overly long multi-fact claims
    if len(tokens) > 32:
        score -= 1.0
    elif len(tokens) > 24:
        score -= 0.5

    # Penalize connective-heavy claims because they often bundle multiple facts
    if re.search(r"\b(and|but|while|after|before|as|which|who)\b", text, flags=re.IGNORECASE):
        score -= 0.2

    return max(score, 0.0)


def _split_long_rule_sentence(sentence: str) -> List[str]:
    """
    Split long sentence fragments using lightweight rules.
    This is only a fallback, not a full parser.
    """
    sentence = _normalize_spaces(sentence)
    context_entity = _pick_context_entity(sentence)
    parts = [sentence]

    # First split on strong punctuation separators
    stage_1 = []
    for part in parts:
        stage_1.extend([p.strip(" ,;") for p in re.split(r"\s*;\s*|\s+--\s+|\s+\-\s+", part) if p.strip(" ,;")])

    # Then split on connectors that often merge multiple news facts
    stage_2 = []
    for part in stage_1:
        split_parts = re.split(
            r"\b(?:but|while|however|after|because|since|following)\b",
            part,
            flags=re.IGNORECASE,
        )
        stage_2.extend([p.strip(" ,;") for p in split_parts if p.strip(" ,;")])

    # Then split on relative clauses and light attribution boundaries.
    stage_3 = []
    for part in stage_2:
        split_parts = re.split(
            r"\s+(?:that|which|who)\s+|,\s*(?:which|who)\s+",
            part,
            flags=re.IGNORECASE,
        )
        stage_3.extend([p.strip(" ,;") for p in split_parts if p.strip(" ,;")])

    # Split comma + and when it joins two event clauses.
    final_parts = []
    for part in stage_3:
        comma_and_parts = [p.strip(" ,;") for p in re.split(r",\s+and\s+", part, flags=re.IGNORECASE) if p.strip(" ,;")]
        if len(comma_and_parts) > 1:
            final_parts.extend(comma_and_parts)
            continue

        if len(part.split()) > 18:
            split_parts = re.split(r"\b(?:and)\b", part, flags=re.IGNORECASE)
            final_parts.extend([p.strip(" ,;") for p in split_parts if p.strip(" ,;")])
        else:
            final_parts.append(part)

    return [
        _replace_vague_subject_with_entity(p, context_entity)
        for p in final_parts
        if p
    ]


def _extract_claims_rule_based_from_sentence(sentence: str) -> List[str]:
    """
    Extract one or more atomic claims from a sentence using rules only.
    """
    sentence = _clean_claim_candidate(sentence)
    if len(sentence.split()) < MIN_SENT_WORDS:
        return []

    if _is_question_or_low_value_sentence(sentence):
        return []

    if not _has_strong_factual_signal(sentence):
        return []

    candidates = []
    fragments = _split_long_rule_sentence(sentence)

    for fragment in fragments:
        fragment = _clean_claim_candidate(fragment)
        if len(fragment.split()) < MIN_CLAIM_WORDS:
            continue
        if _is_question_or_low_value_sentence(fragment):
            continue
        if not _has_strong_factual_signal(fragment):
            continue
        candidates.append(fragment)

    # If splitting removed too much, keep the original sentence
    if not candidates and len(sentence.split()) >= MIN_CLAIM_WORDS:
        candidates = [sentence]

    return _deduplicate_texts(candidates)


# =========================================================
# Seq2seq extraction
# =========================================================
def _get_seq2seq_extractor():
    """
    Lazy-load a text2text-generation pipeline.
    """
    global _SEQ2SEQ_EXTRACTOR, _SEQ2SEQ_LOAD_FAILED

    if _SEQ2SEQ_EXTRACTOR is not None:
        return _SEQ2SEQ_EXTRACTOR

    if _SEQ2SEQ_LOAD_FAILED:
        return None

    if pipeline is None:
        _SEQ2SEQ_LOAD_FAILED = True
        return None

    try:
        _SEQ2SEQ_EXTRACTOR = pipeline(
            "text2text-generation",
            model=CLAIM_EXTRACTOR_MODEL,
            tokenizer=CLAIM_EXTRACTOR_MODEL,
        )
        return _SEQ2SEQ_EXTRACTOR
    except Exception:
        _SEQ2SEQ_LOAD_FAILED = True
        return None


def _build_multi_claim_prompt(sentence: str) -> str:
    """
    Prompt the seq2seq model to output multiple atomic claims.
    """
    return (
        "Extract all atomic verifiable factual claims from the news sentence.\n"
        "Rules:\n"
        "- Split combined statements into separate claims.\n"
        "- Keep named entities, numbers, dates, locations, and core events.\n"
        "- Remove subjective, emotional, or evaluative language.\n"
        "- Rewrite each claim in a neutral factual style.\n"
        "- Output one claim per line.\n"
        "- If there is no verifiable factual claim, output exactly: NO_CLAIM\n\n"
        f"Sentence: {sentence}\n\n"
        "Claims:\n"
    )


def _build_retrieval_summary_prompt(passage: str) -> str:
    """
    Prompt the seq2seq model to compress a passage into retrieval-ready factual summaries.
    """
    return (
        "Summarize the news passage into short retrieval-ready factual statements.\n"
        "Rules:\n"
        "- Keep only verifiable facts.\n"
        "- Preserve named entities, numbers, dates, locations, and outcomes.\n"
        "- Each line must be a complete standalone sentence with an explicit subject.\n"
        "- Make each line easy to search.\n"
        "- Prefer concise neutral wording.\n"
        "- Do not output fragments, headlines, or descriptive phrases.\n"
        "- Avoid vague pronouns like 'it' or 'they' unless the referent is explicit in the same line.\n"
        f"- Output at most {MAX_RETRIEVAL_SUMMARIES_PER_CHUNK} lines.\n"
        "- If there is no verifiable factual content, output exactly: NO_CLAIM\n\n"
        f"Passage: {passage}\n\n"
        "Retrieval summaries:\n"
    )


def _clean_model_output_line(text: str) -> str:
    """
    Clean a single line generated by the model.
    """
    text = str(text).strip()

    text = re.sub(r"^(claim|claims|output|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\d+[\).\-\:]\s*", "", text).strip()
    text = re.sub(r"^[-*•]\s*", "", text).strip()

    text = _clean_claim_candidate(text)

    # Remove surrounding quotes
    text = text.strip("\"'“”‘’").strip()

    return text


def _parse_multi_claim_output(generated_text: str) -> List[str]:
    """
    Parse model output into a list of claim strings.
    """
    generated_text = str(generated_text).strip()

    if not generated_text:
        return []

    if generated_text.lower().strip() in {"no_claim", "no claim", "none", "n/a", "not a claim"}:
        return []

    # Split by newline first
    raw_lines = [line.strip() for line in generated_text.splitlines() if line.strip()]

    # If the model returned a single paragraph, try splitting it conservatively
    if len(raw_lines) == 1:
        single = raw_lines[0]
        if ";" in single:
            raw_lines = [x.strip() for x in single.split(";") if x.strip()]
        else:
            raw_lines = [single]

    claims = []
    for line in raw_lines:
        cleaned = _clean_model_output_line(line)
        if not cleaned:
            continue

        if cleaned.lower() in {"no_claim", "no claim", "none", "n/a", "not a claim"}:
            continue

        if len(cleaned.split()) < MIN_CLAIM_WORDS:
            continue

        if _is_question_or_low_value_sentence(cleaned):
            continue

        if not _has_strong_factual_signal(cleaned):
            continue

        claims.append(cleaned)

    claims = _deduplicate_texts(claims)
    return claims[:MAX_CLAIMS_PER_SENTENCE]


def _extract_claims_model_based_from_sentence(sentence: str) -> List[str]:
    """
    Use a seq2seq model to extract multiple atomic claims from one sentence.
    """
    extractor = _get_seq2seq_extractor()
    if extractor is None:
        return []

    sentence = _clean_claim_candidate(sentence)
    if len(sentence.split()) < MIN_SENT_WORDS:
        return []

    if _is_question_or_low_value_sentence(sentence):
        return []

    if len(sentence) > MAX_INPUT_CHARS_PER_SENT:
        sentence = sentence[:MAX_INPUT_CHARS_PER_SENT].rsplit(" ", 1)[0].strip()

    prompt = _build_multi_claim_prompt(sentence)

    try:
        outputs = extractor(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            truncation=True,
        )
        if not outputs:
            return []

        generated_text = outputs[0].get("generated_text", "")
        claims = _parse_multi_claim_output(generated_text)

        # If the model returned nothing, let the caller fall back to rules
        return claims
    except Exception:
        return []


def _extract_retrieval_summaries_from_chunk(chunk_text: str) -> List[str]:
    """
    Use a seq2seq model to turn a passage chunk into search-friendly factual summaries.
    """
    extractor = _get_seq2seq_extractor()
    if extractor is None:
        return []

    chunk_text = clean_claim_text(chunk_text)
    chunk_text = _normalize_spaces(chunk_text)

    if len(chunk_text.split()) < MIN_SENT_WORDS:
        return []

    if len(chunk_text) > MAX_INPUT_CHARS_PER_SUMMARY_CHUNK:
        chunk_text = chunk_text[:MAX_INPUT_CHARS_PER_SUMMARY_CHUNK].rsplit(" ", 1)[0].strip()

    prompt = _build_retrieval_summary_prompt(chunk_text)

    try:
        outputs = extractor(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            truncation=True,
        )
        if not outputs:
            return []

        generated_text = outputs[0].get("generated_text", "")
        claims = _parse_multi_claim_output(generated_text)
        return claims[:MAX_RETRIEVAL_SUMMARIES_PER_CHUNK]
    except Exception:
        return []


def _extract_claims_model_based_from_document(article_text: str) -> List[Dict]:
    """
    Convert a long article into retrieval-ready factual summaries using chunked seq2seq prompting.
    """
    extractor = _get_seq2seq_extractor()
    if extractor is None:
        return []

    article_text = clean_claim_text(article_text)
    article_text = _normalize_spaces(article_text)
    if len(article_text.split()) < MIN_SENT_WORDS:
        return []

    chunk_candidates = []
    for chunk_idx, chunk_text in enumerate(_chunk_article_for_summary(article_text), start=1):
        summaries = _extract_retrieval_summaries_from_chunk(chunk_text)
        for summary in summaries:
            summary = _clean_claim_candidate(summary)
            if not _claim_is_worth_verifying(summary):
                continue

            score = _score_atomic_claim(summary) + 0.6
            chunk_candidates.append(
                {
                    "text": summary,
                    "score": round(score, 4),
                    "source_sentence": chunk_text,
                    "source_sentence_id": chunk_idx,
                    "method": "summary_model",
                }
            )

    if not chunk_candidates:
        return []

    chunk_candidates = _deduplicate_claim_dicts(chunk_candidates)
    chunk_candidates = sorted(chunk_candidates, key=lambda x: x["score"], reverse=True)
    return chunk_candidates


# =========================================================
# Post-processing and filtering
# =========================================================
def _claim_is_worth_verifying(text: str) -> bool:
    """
    Final gate to keep only claims that are likely useful for retrieval and verification.
    """
    text = _clean_claim_candidate(text)
    tokens = text.split()

    if len(tokens) < MIN_CLAIM_WORDS:
        return False

    if len(tokens) > MAX_CLAIM_WORDS:
        return False

    if _looks_like_fragment(text):
        return False

    if _is_question_or_low_value_sentence(text):
        return False

    if _contains_subjective_language(text):
        return False

    if _starts_with_vague_pronoun(text) and not _has_specific_search_anchor(text):
        return False

    if not _has_strong_factual_signal(text):
        return False

    if not _has_specific_search_anchor(text):
        return False

    # Drop highly vague action descriptions that are usually hard to verify in news search
    if re.search(
        r"\b(looked for|tried to|aimed to|hoped to|appeared to|seemed to|were on the back foot)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False

    return True


def _deduplicate_texts(texts: List[str]) -> List[str]:
    """
    Deduplicate claim texts using Jaccard similarity.
    """
    kept = []
    for text in texts:
        duplicate = False
        for prev in kept:
            if _jaccard_similarity(text, prev) >= DEDUP_SIM_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(text)
    return kept


def _deduplicate_claim_dicts(claims: List[Dict]) -> List[Dict]:
    """
    Deduplicate claim dictionaries while keeping higher-score entries first.
    """
    deduped = []
    for claim in sorted(claims, key=lambda x: x["score"], reverse=True):
        duplicate = False
        for kept in deduped:
            if _jaccard_similarity(claim["text"], kept["text"]) >= DEDUP_SIM_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            deduped.append(claim)
    return deduped


def _score_atomic_claim(text: str) -> float:
    """
    Score an already-extracted atomic claim.
    """
    score = _rule_claim_score(text)

    # Reward concise, retrieval-friendly claims
    token_len = len(text.split())
    if 5 <= token_len <= 14:
        score += 0.8
    elif 15 <= token_len <= 20:
        score += 0.3

    if _extract_numbers(text):
        score += 0.3

    if _extract_capitalized_entities(text):
        score += 0.3

    return round(score, 4)


def _make_claim_entry(
    claim_text: str,
    score_bonus: float,
    source_sentence: str,
    source_sentence_id: int,
    method: str,
) -> Dict:
    score = _score_atomic_claim(claim_text) + score_bonus
    return {
        "claim_id": 0,
        "text": claim_text,
        "score": round(score, 4),
        "source_sentence": source_sentence,
        "source_sentence_id": source_sentence_id,
        "method": method,
    }


def _finalize_claim_candidates(candidates: List[Dict], top_k: int) -> List[Dict]:
    if not candidates:
        return []

    finalized = _deduplicate_claim_dicts(candidates)
    finalized = sorted(finalized, key=lambda x: x["score"], reverse=True)[:top_k]

    results = []
    for idx, item in enumerate(finalized, start=1):
        updated = dict(item)
        updated["claim_id"] = idx
        results.append(updated)

    return results


def extract_claims_progressively(article_text: str, top_k: int = TOP_K_CLAIMS):
    """
    Yield intermediate extraction results so the UI can show progress chunk by chunk.
    """
    article_text = clean_claim_text(article_text)
    article_text = _normalize_spaces(article_text)
    sentences = split_sentences(article_text)

    if CLAIM_EXTRACTOR_MODE in {"model", "summary", "summary_model"}:
        running_candidates = []
        chunks = _chunk_article_for_summary(article_text)

        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            chunk_candidates = []
            summaries = _extract_retrieval_summaries_from_chunk(chunk_text)

            for summary in summaries:
                summary = _clean_claim_candidate(summary)
                if not _claim_is_worth_verifying(summary):
                    continue

                item = _make_claim_entry(
                    claim_text=summary,
                    score_bonus=0.6,
                    source_sentence=chunk_text,
                    source_sentence_id=chunk_idx,
                    method="summary_model",
                )
                chunk_candidates.append(item)
                running_candidates.append(item)

            yield {
                "stage": "summary_chunk",
                "chunk_id": chunk_idx,
                "total_chunks": len(chunks),
                "chunk_text": chunk_text,
                "chunk_claims": _finalize_claim_candidates(
                    chunk_candidates,
                    MAX_RETRIEVAL_SUMMARIES_PER_CHUNK,
                ),
                "current_claims": _finalize_claim_candidates(running_candidates, top_k),
            }

        final_claims = _finalize_claim_candidates(running_candidates, top_k)
        if final_claims:
            yield {
                "stage": "done",
                "current_claims": final_claims,
            }
            return

    running_candidates = []
    context_entity = None
    usable_sentences = [
        (sent_idx + 1, _normalize_spaces(sentence))
        for sent_idx, sentence in enumerate(sentences)
        if len(_normalize_spaces(sentence).split()) >= MIN_SENT_WORDS
        and not _is_question_or_low_value_sentence(_normalize_spaces(sentence))
    ]

    for idx, (sent_id, source_sentence) in enumerate(usable_sentences, start=1):
        source_sentence = _replace_vague_subject_with_entity(source_sentence, context_entity)
        sentence_candidates = []
        for claim_text in _extract_claims_rule_based_from_sentence(source_sentence):
            claim_text = _clean_claim_candidate(claim_text)
            if not _claim_is_worth_verifying(claim_text):
                continue

            item = _make_claim_entry(
                claim_text=claim_text,
                score_bonus=0.0,
                source_sentence=source_sentence,
                source_sentence_id=sent_id,
                method="rule",
            )
            sentence_candidates.append(item)
            running_candidates.append(item)

        if sentence_candidates:
            yield {
                "stage": "rule_sentence",
                "chunk_id": idx,
                "total_chunks": len(usable_sentences),
                "chunk_text": source_sentence,
                "chunk_claims": _finalize_claim_candidates(
                    sentence_candidates,
                    MAX_CLAIMS_PER_SENTENCE,
                ),
                "current_claims": _finalize_claim_candidates(running_candidates, top_k),
            }

        updated_context_entity = _pick_context_entity(source_sentence)
        context_entity = _prefer_context_entity(context_entity, updated_context_entity)

    yield {
        "stage": "done",
        "current_claims": _finalize_claim_candidates(running_candidates, top_k),
    }


# =========================================================
# Public API
# =========================================================
def extract_claims(article_text: str, top_k: int = TOP_K_CLAIMS) -> List[Dict]:
    """
    Extract claim dictionaries from an article.

    Output format remains compatible with the current app.py:
    [
        {
            "claim_id": 1,
            "text": "...",
            "score": 4.3,
            "source_sentence": "...",
            "method": "model" or "rule"
        },
        ...
    ]
    """
    latest_claims = []
    for event in extract_claims_progressively(article_text, top_k=top_k):
        latest_claims = event.get("current_claims", latest_claims)
    return latest_claims[:top_k]
