import os
import re
import time
import requests
from typing import List, Dict

from config import GNEWS_API_KEY, GNEWS_ENDPOINT, TOP_K_EVIDENCE, REQUEST_TIMEOUT


MAX_QUERY_CANDIDATES = int(os.getenv("MAX_QUERY_CANDIDATES", "3"))
EARLY_STOP_ARTICLE_COUNT = int(os.getenv("EARLY_STOP_ARTICLE_COUNT", str(max(TOP_K_EVIDENCE * 2, 6))))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def _token_overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _clean_keyword(token: str) -> str:
    """
    Keep only letters, digits, and percent signs.
    Remove punctuation such as hyphens, quotes, slashes, etc.
    """
    token = str(token)
    token = re.sub(r"[^A-Za-z0-9%]", "", token)
    return token.strip()


def _extract_numbers(text: str) -> List[str]:
    """
    Extract numbers, percentages, and years.
    Examples: 10, 10.5, 10%, 2024
    """
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text))


def _extract_capitalized_entities(text: str) -> List[str]:
    """
    Extract simple capitalized entity-like tokens.
    Examples: Tesla, Apple, China, Biden
    """
    blocked_singletons = {
        "A", "An", "The", "It", "This", "That", "These", "Those",
        "He", "She", "They", "We", "You", "I",
        "Previously", "Currently", "Formerly", "Meanwhile", "However", "Later",
    }
    spans = re.findall(r"\b[A-Z][a-zA-Z0-9&.\-]+(?:\s+[A-Z][a-zA-Z0-9&.\-]+){0,3}\b", str(text))
    deduped = []
    seen = set()
    for span in spans:
        if span in blocked_singletons:
            continue
        key = span.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(span)
    return deduped


def _match_ratio(items: List[str], text: str) -> float:
    """
    Compute the fraction of items that appear in the given text.
    """
    if not items:
        return 0.0

    text_lower = str(text).lower()
    hits = 0
    for item in items:
        if str(item).lower() in text_lower:
            hits += 1

    return hits / len(items)


def _extract_keywords(claim: str, max_terms: int = 10) -> List[str]:
    """
    Extract search-friendly keywords from a claim.

    Improvements over the old version:
    1. Preserve more entities, numbers, and event/action words.
    2. Do not over-remove important news verbs such as
       'announced', 'confirmed', 'denied', etc.
    """
    raw_words = re.findall(r"\b[\w\-']+\b", str(claim))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
        "that", "this", "these", "those", "it", "its", "he", "she", "they",
        "his", "her", "their", "and", "or", "but", "if", "then", "than",
        "just", "soon", "after", "before", "local", "media", "following",
        "update", "posted", "further", "popular", "first", "people",
        "had", "have", "has", "may", "might", "will", "can", "cannot", "up",
        "very", "more", "most", "over", "under", "into", "onto", "also",
        "such", "some", "many", "much", "few", "several",
        "yesterday", "today", "tomorrow",
        "previously", "currently", "formerly", "meanwhile", "however", "later",
    }

    keywords = []
    for word in raw_words:
        cleaned = _clean_keyword(word)
        if not cleaned:
            continue

        lowered = cleaned.lower()
        if lowered in stopwords:
            continue

        # Keep:
        # 1) numbers / percentages
        # 2) capitalized words (likely entities)
        # 3) longer words
        # 4) common event/action words in news
        if (
            re.fullmatch(r"\d+(?:\.\d+)?%?", cleaned)
            or word[:1].isupper()
            or len(cleaned) >= 4
            or lowered in {
                "announced", "confirmed", "denied", "approved", "reduced",
                "launched", "warned", "revealed", "claimed", "reported",
                "stated", "showed", "signed", "banned", "arrested", "charged",
                "cut", "cuts", "raised", "fell", "fall", "wins", "won"
            }
        ):
            keywords.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for keyword in keywords:
        lowered = keyword.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(keyword)

    return deduped[:max_terms]


def _is_bad_query(query: str) -> bool:
    """
    Detect low-quality queries that are not worth sending to GNews.
    """
    if not query or len(query.strip()) < 3:
        return True

    tokens = query.split()
    if not tokens:
        return True

    # If most tokens are numeric, the query is probably malformed
    digit_tokens = sum(
        1 for token in tokens
        if re.fullmatch(r"\d+(?:\.\d+)?%?", token)
    )
    if digit_tokens / max(len(tokens), 1) > 0.6:
        return True

    # Single-token short queries are usually too weak
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return True

    return False


def build_query_candidates(claim: str) -> List[str]:
    """
    Generate multiple search queries for the same claim.

    Improvements:
    1. Do not rely only on "long keywords -> short keywords".
    2. Add multiple retrieval views:
       - cleaned full claim
       - entity + keywords
       - keyword-rich query
       - short core query
       - entity + numbers
       - non-numeric query
    3. Prepare for multi-query retrieval and merge.
    """
    keywords = _extract_keywords(claim, max_terms=10)
    entities = _extract_capitalized_entities(claim)
    numbers = _extract_numbers(claim)

    candidates = []

    # 1) Cleaned full claim
    raw = re.sub(r"[^A-Za-z0-9\s%]", " ", str(claim))
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(
        r"^(previously|currently|formerly|reportedly|allegedly|meanwhile|however|later)\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    if raw:
        candidates.append(raw[:120])

    # 2) Entities + keywords
    if entities and keywords:
        candidates.append(" ".join((entities[:2] + keywords[:4])[:6]))

    # 3) Full keyword-rich query
    if keywords:
        candidates.append(" ".join(keywords[:8]))

    # 4) Short core query
    if len(keywords) >= 4:
        candidates.append(" ".join(keywords[:4]))
    elif len(keywords) >= 2:
        candidates.append(" ".join(keywords[:2]))

    # 5) Entities + numbers
    if entities and numbers:
        candidates.append(" ".join((entities[:2] + numbers[:2])[:4]))

    # 6) Non-numeric keyword query
    non_numeric_keywords = [
        keyword for keyword in keywords
        if not re.fullmatch(r"\d+(?:\.\d+)?%?", keyword)
    ]
    if len(non_numeric_keywords) >= 5:
        candidates.append(" ".join(non_numeric_keywords[:5]))
    elif len(non_numeric_keywords) >= 3:
        candidates.append(" ".join(non_numeric_keywords[:3]))

    # Deduplicate + filter out poor queries
    final_candidates = []
    seen = set()
    for query in candidates:
        query = re.sub(r"\s+", " ", query).strip()
        lowered = query.lower()
        if not query or lowered in seen:
            continue
        if _is_bad_query(query):
            continue
        seen.add(lowered)
        final_candidates.append(query)

    return final_candidates[:MAX_QUERY_CANDIDATES]


def _format_gnews_articles(articles: List[Dict], claim: str) -> List[Dict]:
    """
    Re-rank retrieved GNews articles locally.

    Problems in the old version:
    - Mostly relied on token overlap
    - Over-relied on original GNews ranking order

    New version adds:
    - title_overlap_score
    - text_overlap_score
    - entity_match_score
    - number_match_score
    - matched_entities
    - matched_numbers
    """
    results = []

    claim_entities = _extract_capitalized_entities(claim)
    claim_numbers = _extract_numbers(claim)

    for idx, article in enumerate(articles):
        source_info = article.get("source", {}) or {}
        title = article.get("title", "") or ""
        text = article.get("description", "") or article.get("content", "") or ""
        combined_text = f"{title}. {text}".strip()

        title_overlap = _token_overlap_ratio(claim, title)
        text_overlap = _token_overlap_ratio(claim, text)
        entity_match = _match_ratio(claim_entities, combined_text)
        number_match = _match_ratio(claim_numbers, combined_text)
        api_rank_score = 1.0 / (idx + 1)

        # Source credibility is intentionally not included yet
        final_score = (
            0.30 * title_overlap
            + 0.25 * text_overlap
            + 0.20 * entity_match
            + 0.15 * number_match
            + 0.10 * api_rank_score
        )

        matched_entities = [
            entity for entity in claim_entities
            if entity.lower() in combined_text.lower()
        ]
        matched_numbers = [
            number for number in claim_numbers
            if number.lower() in combined_text.lower()
        ]

        results.append(
            {
                "evidence_id": idx + 1,
                "title": title,
                "text": text,
                "combined_text": combined_text,
                "source": source_info.get("name", "Unknown"),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "api_rank_score": round(api_rank_score, 4),
                "title_overlap_score": round(title_overlap, 4),
                "text_overlap_score": round(text_overlap, 4),
                "entity_match_score": round(entity_match, 4),
                "number_match_score": round(number_match, 4),
                "matched_entities": matched_entities,
                "matched_numbers": matched_numbers,
                "score": round(final_score, 4),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _search_once(query: str, top_k: int) -> Dict:
    params = {
        "q": query,
        "max": top_k,
        "apikey": GNEWS_API_KEY,
        "lang": "en",
        "sortby": "relevance",
        "in": "title,description",
    }

    response = requests.get(GNEWS_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)

    # Respect GNews free-tier rate limits
    time.sleep(1.05)

    if response.status_code != 200:
        try:
            error_payload = response.json()
        except Exception:
            error_payload = response.text

        return {
            "ok": False,
            "articles": [],
            "error": f"GNews API error {response.status_code}: {error_payload}",
        }

    data = response.json()
    return {
        "ok": True,
        "articles": data.get("articles", []),
        "error": None,
    }


def retrieve_evidence(claim: str, top_k: int = TOP_K_EVIDENCE) -> Dict:
    """
    Return format:
    {
        "query": "...",           # all search queries actually used
        "results": [...],         # ranked evidence list
        "error": None or "..."
    }

    Core improvements:
    1. Do not stop after the first query that returns results
    2. Search with multiple queries
    3. Merge all retrieved articles
    4. Deduplicate by URL
    5. Re-rank globally and return top_k
    """
    if not GNEWS_API_KEY:
        return {
            "query": "",
            "results": [],
            "error": "GNEWS_API_KEY is missing. Please check your .env file.",
        }

    query_candidates = build_query_candidates(claim)

    if not query_candidates:
        return {
            "query": "",
            "results": [],
            "error": "No valid search query could be built from this claim.",
        }

    all_articles = []
    used_queries = []
    last_error = None

    for query in query_candidates:
        search_result = _search_once(query, top_k=top_k)
        used_queries.append(query)

        if not search_result["ok"]:
            last_error = search_result["error"]
            continue

        articles = search_result["articles"]
        if articles:
            all_articles.extend(articles)

        seen_keys = set()
        for article in all_articles:
            url = str(article.get("url", "")).strip()
            title = str(article.get("title", "")).strip()
            published_at = str(article.get("publishedAt", "")).strip()
            seen_keys.add(url if url else f"{title}__{published_at}")

        if len(seen_keys) >= EARLY_STOP_ARTICLE_COUNT:
            break

    if not all_articles:
        return {
            "query": " | ".join(query_candidates),
            "results": [],
            "error": last_error,
        }

    # Deduplicate: prefer URL, otherwise fall back to title + published_at
    deduplicated = {}
    for article in all_articles:
        url = str(article.get("url", "")).strip()
        title = str(article.get("title", "")).strip()
        published_at = str(article.get("publishedAt", "")).strip()

        key = url if url else f"{title}__{published_at}"
        if key not in deduplicated:
            deduplicated[key] = article

    ranked_results = _format_gnews_articles(list(deduplicated.values()), claim)

    return {
        "query": " | ".join(used_queries),
        "results": ranked_results[:top_k],
        "error": None,
    }
