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
    token = str(token)
    token = re.sub(r"[^A-Za-z0-9%]", "", token)
    return token.strip()


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text))


def _extract_capitalized_entities(text: str) -> List[str]:
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
    if not items:
        return 0.0
    text_lower = str(text).lower()
    hits = sum(1 for item in items if str(item).lower() in text_lower)
    return hits / len(items)


def _extract_keywords(claim: str, max_terms: int = 10) -> List[str]:
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

    seen = set()
    deduped = []
    for keyword in keywords:
        lowered = keyword.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(keyword)

    return deduped[:max_terms]


def _is_bad_query(query: str) -> bool:
    if not query or len(query.strip()) < 3:
        return True
    tokens = query.split()
    if not tokens:
        return True
    digit_tokens = sum(1 for token in tokens if re.fullmatch(r"\d+(?:\.\d+)?%?", token))
    if digit_tokens / max(len(tokens), 1) > 0.6:
        return True
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return True
    return False


def build_query_candidates(claim: str) -> List[str]:
    keywords = _extract_keywords(claim, max_terms=10)
    entities = _extract_capitalized_entities(claim)
    numbers = _extract_numbers(claim)

    candidates = []

    raw = re.sub(r"[^A-Za-z0-9\s%]", " ", str(claim))
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(
        r"^(previously|currently|formerly|reportedly|allegedly|meanwhile|however|later)\s+",
        "", raw, flags=re.IGNORECASE,
    )
    if raw:
        raw_words = raw.split()
        if len(raw_words) > 8:
            raw = " ".join(raw_words[:8])
        candidates.append(raw)

    if entities and keywords:
        candidates.append(" ".join((entities[:2] + keywords[:4])[:6]))

    if keywords:
        candidates.append(" ".join(keywords[:8]))

    if len(keywords) >= 4:
        candidates.append(" ".join(keywords[:4]))
    elif len(keywords) >= 2:
        candidates.append(" ".join(keywords[:2]))

    if entities and numbers:
        candidates.append(" ".join((entities[:2] + numbers[:2])[:4]))

    non_numeric_keywords = [
        kw for kw in keywords if not re.fullmatch(r"\d+(?:\.\d+)?%?", kw)
    ]
    if len(non_numeric_keywords) >= 5:
        candidates.append(" ".join(non_numeric_keywords[:5]))
    elif len(non_numeric_keywords) >= 3:
        candidates.append(" ".join(non_numeric_keywords[:3]))

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
    results = []
    claim_entities = _extract_capitalized_entities(claim)
    claim_numbers = _extract_numbers(claim)

    for idx, article in enumerate(articles):
        source_info = article.get("source", {}) or {}
        title = (article.get("title", "") or "")[:300]
        text = (article.get("description", "") or article.get("content", "") or "")[:1000]
        combined_text = f"{title}. {text}".strip()

        title_overlap = _token_overlap_ratio(claim, title)
        text_overlap = _token_overlap_ratio(claim, text)
        entity_match = _match_ratio(claim_entities, combined_text)
        number_match = _match_ratio(claim_numbers, combined_text)
        api_rank_score = 1.0 / (idx + 1)

        final_score = (
            0.30 * title_overlap
            + 0.25 * text_overlap
            + 0.20 * entity_match
            + 0.15 * number_match
            + 0.10 * api_rank_score
        )

        matched_entities = [e for e in claim_entities if e.lower() in combined_text.lower()]
        matched_numbers = [n for n in claim_numbers if n.lower() in combined_text.lower()]

        results.append({
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
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


_MAX_RETRIES = int(os.getenv("GNEWS_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.getenv("GNEWS_RETRY_BASE_DELAY", "2.0"))  # seconds


def _search_once(query: str, top_k: int) -> Dict:
    params = {
        "q": query,
        "max": top_k,
        "apikey": GNEWS_API_KEY,
        "lang": "en",
        "sortby": "relevance",
        "in": "title,description",
    }

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.get(GNEWS_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            last_error = "Request timed out."
            time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
            continue
        except requests.exceptions.RequestException as exc:
            return {"ok": False, "articles": [], "error": str(exc)}

        # GNews free tier rate limit — always wait between calls
        time.sleep(1.05)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", _RETRY_BASE_DELAY * (2 ** attempt)))
            last_error = f"GNews rate limit hit (429). Waiting {retry_after}s before retry."
            print(f"[retriever] {last_error}")
            time.sleep(retry_after)
            continue

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

    return {"ok": False, "articles": [], "error": last_error or "Max retries exceeded."}


def _article_key(article: Dict) -> str:
    url = str(article.get("url", "")).strip()
    title = str(article.get("title", "")).strip()
    published_at = str(article.get("publishedAt", "")).strip()
    return url if url else f"{title}__{published_at}"


def retrieve_evidence(claim: str, top_k: int = TOP_K_EVIDENCE) -> Dict:
    """
    Return format:
    {
        "query": "...",           # all search queries actually used
        "results": [...],         # ranked evidence list
        "error": None or "..."
    }
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
    seen_keys = set()          # ← 移到循环外，跨 query 累积去重

    for query in query_candidates:
        search_result = _search_once(query, top_k=top_k)
        used_queries.append(query)

        if not search_result["ok"]:
            last_error = search_result["error"]
            continue

        for article in search_result["articles"]:
            key = _article_key(article)
            if key not in seen_keys:
                seen_keys.add(key)
                all_articles.append(article)

        if len(seen_keys) >= EARLY_STOP_ARTICLE_COUNT:
            break

    if not all_articles:
        return {
            "query": " | ".join(query_candidates),
            "results": [],
            "error": last_error,
        }

    ranked_results = _format_gnews_articles(all_articles, claim)

    return {
        "query": " | ".join(used_queries),
        "results": ranked_results[:top_k],
        "error": None,
    }
