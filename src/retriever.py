import re
import time
import requests
from typing import Any, List, Dict

from src.config import (
    EARLY_STOP_ARTICLE_COUNT,
    EVIDENCE_CACHE_MAX_SIZE,
    GNEWS_API_KEY,
    GNEWS_DIRECT_SENTENCE_QUERY,
    GNEWS_ENDPOINT,
    GNEWS_MAX_RETRIES,
    GNEWS_RESULTS_PER_QUERY,
    GNEWS_RETRY_BASE_DELAY,
    GNEWS_SEARCH_FIELDS,
    GNEWS_SORTBY_ORDER,
    MAX_GNEWS_QUERY_CHARS,
    MAX_GNEWS_QUERY_WORDS,
    MAX_QUERY_CANDIDATES,
    MIN_EVIDENCE_SCORE,
    MIN_KEYWORD_MATCH,
    REQUEST_TIMEOUT,
    TOP_K_EVIDENCE,
)


_RERANK_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
    "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "his", "her", "their", "and", "or", "but", "if", "then", "than",
    "said", "says", "say", "according", "reported", "reportedly",
    "new", "latest", "week", "month", "year", "today", "yesterday",
    "tomorrow", "news", "update", "local", "media",
}

_ENTITY_ALIASES = {
    "us": ["us", "u s", "u.s", "u.s.", "united states", "american"],
    "u s": ["us", "u s", "u.s", "u.s.", "united states", "american"],
    "u.s.": ["us", "u s", "u.s", "u.s.", "united states", "american"],
    "centcom": ["centcom", "central command", "us central command", "u.s. central command"],
    "uk": ["uk", "u.k.", "united kingdom", "british"],
}

# ── Source credibility scores (0.0 – 1.0) ────────────────────────────────────
# Tier 1 (0.9+): major wire services and public broadcasters
# Tier 2 (0.7–0.89): established national newspapers
# Tier 3 (0.5–0.69): regional / mixed-reliability outlets
# Unknown sources default to 0.4
_SOURCE_CREDIBILITY: dict = {
    # Wire services
    "reuters": 0.95, "associated press": 0.95, "ap news": 0.95,
    "afp": 0.93, "bloomberg": 0.92,
    # Public broadcasters
    "bbc": 0.92, "bbc news": 0.92, "npr": 0.91,
    "abc news": 0.88, "cbs news": 0.88, "nbc news": 0.88,
    "pbs": 0.90, "channel 4": 0.88,

    # Major newspapers
    "the new york times": 0.90, "new york times": 0.90,
    "the washington post": 0.89, "washington post": 0.89,
    "the guardian": 0.88, "guardian": 0.88,
    "the wall street journal": 0.89, "wall street journal": 0.89,
    "financial times": 0.90, "the economist": 0.90,
    "los angeles times": 0.87, "chicago tribune": 0.85,
    "the times": 0.87, "the telegraph": 0.85,
    "le monde": 0.88, "der spiegel": 0.88,
    # Tech / science
    "nature": 0.95, "science": 0.95, "mit technology review": 0.90,
    "wired": 0.82, "ars technica": 0.83, "the verge": 0.80,
    # Mixed / lower tier
    "cnn": 0.78, "fox news": 0.65, "daily mail": 0.55,
    "buzzfeed news": 0.70, "huffpost": 0.72,
    "indiatimes": 0.60, "ndtv": 0.72, "the hindu": 0.82,
    "al jazeera": 0.82, "south china morning post": 0.78,
    # Singapore media
    "cna": 0.88, "channel news asia": 0.88, "the straits times": 0.88, "straitstimes": 0.88,
}
_SOURCE_CREDIBILITY_DEFAULT = 0.40

def _get_source_credibility(source_name: str) -> float:
    """Return a credibility score for the given source name."""
    key = str(source_name).lower().strip()
    # Exact match first
    if key in _SOURCE_CREDIBILITY:
        return _SOURCE_CREDIBILITY[key]
    # Partial match (e.g. "BBC News UK" -> "bbc news")
    for known, score in _SOURCE_CREDIBILITY.items():
        if known in key or key in known:
            return score
    return _SOURCE_CREDIBILITY_DEFAULT


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def _content_tokens(text: str) -> List[str]:
    return [
        token
        for token in _tokenize(text)
        if token not in _RERANK_STOPWORDS and len(token) > 2
    ]


def _token_overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(_content_tokens(a))
    tokens_b = set(_content_tokens(b))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _normalize_for_match(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\bu\.s\.", "us", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _term_in_text(term: str, text: str) -> bool:
    term = _normalize_for_match(term)
    text = _normalize_for_match(text)
    if not term:
        return False
    aliases = _ENTITY_ALIASES.get(term, [term])
    return any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases)


def _match_items(items: List[str], text: str) -> List[str]:
    return [item for item in items if _term_in_text(item, text)]


def _clean_keyword(token: str) -> str:
    token = str(token)
    token = re.sub(r"([A-Z]{2,})([a-z])", r"\1 \2", token)
    token = re.sub(r"[^A-Za-z0-9\s]", " ", token)
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def _sanitize_gnews_query(query: str) -> str:
    """
    Convert a claim fragment into simple GNews keywords.

    GNews parses punctuation as query syntax, so strip special characters before
    sending q=... to avoid 400 syntax errors.
    """
    query = str(query).replace("%", " percent ")
    query = re.sub(r"[^A-Za-z0-9\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(r"\bU\s+S\b", "US", query, flags=re.IGNORECASE)

    words = [
        word
        for word in query.split()
        if word.lower() not in {"and", "or", "not"}
    ]
    if len(words) > MAX_GNEWS_QUERY_WORDS:
        words = words[:MAX_GNEWS_QUERY_WORDS]

    return " ".join(words)


def _sanitize_gnews_sentence_query(query: str) -> str:
    """
    Keep the original sentence order for GNews while removing query syntax chars.

    GNews q has a length limit and treats punctuation/operators specially, so this
    is "original sentence after safe cleanup", not keyword extraction.
    """
    query = str(query).replace("%", " percent ")
    query = re.sub(r"[^A-Za-z0-9\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(r"\bU\s+S\b", "US", query, flags=re.IGNORECASE)

    words = [
        word
        for word in query.split()
        if word.lower() not in {"and", "or", "not"}
    ]
    cleaned = " ".join(words)
    if len(cleaned) <= MAX_GNEWS_QUERY_CHARS:
        return cleaned

    trimmed_words = []
    current_len = 0
    for word in words:
        next_len = len(word) if not trimmed_words else current_len + 1 + len(word)
        if next_len > MAX_GNEWS_QUERY_CHARS:
            break
        trimmed_words.append(word)
        current_len = next_len
    return " ".join(trimmed_words)


def _dedupe_terms(terms: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for term in terms:
        cleaned = _sanitize_gnews_query(term)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(cleaned)
    return deduped


def _exclude_entity_terms(keywords: List[str], entities: List[str]) -> List[str]:
    entity_words = set()
    for entity in entities:
        entity_words.update(_sanitize_gnews_query(entity).lower().split())
    cleaned_keywords = []
    for kw in keywords:
        terms = [
            term
            for term in _sanitize_gnews_query(kw).split()
            if term.lower() not in entity_words
        ]
        cleaned = " ".join(terms)
        if cleaned:
            cleaned_keywords.append(cleaned)
    return cleaned_keywords


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text))


def _extract_capitalized_entities(text: str) -> List[str]:
    blocked_singletons = {
        "A", "An", "The", "It", "This", "That", "These", "Those",
        "He", "She", "They", "We", "You", "I", "In", "On", "At",
        "Previously", "Currently", "Formerly", "Meanwhile", "However", "Later",
    }
    text = re.sub(r"\bU\.S\.", "US", str(text), flags=re.IGNORECASE)
    spans = re.findall(r"\b[A-Z][a-zA-Z0-9&.\-]+(?:\s+[A-Z][a-zA-Z0-9&.\-]+){0,3}\b", text)
    deduped = []
    seen = set()
    for span in spans:
        parts = span.split()
        while parts and parts[0] in blocked_singletons:
            parts = parts[1:]
        span = " ".join(parts)
        if not span or span in blocked_singletons:
            continue
        normalized = _sanitize_gnews_query(span)
        if normalized.lower() == "us flagged":
            normalized = "US"
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _match_ratio(items: List[str], text: str) -> float:
    if not items:
        return 0.0
    hits = sum(1 for item in items if _term_in_text(item, text))
    return hits / len(items)


def _extract_keywords(claim: str, max_terms: int = 10) -> List[str]:
    claim = re.sub(r"\bU\.S\.", "US", str(claim), flags=re.IGNORECASE)
    raw_words = re.findall(r"\b[\w\-']+\b", claim)

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
        "that", "this", "these", "those", "it", "its", "he", "she", "they",
        "his", "her", "their", "and", "or", "but", "if", "then", "than",
        "just", "soon", "after", "before", "local", "media", "following",
        "update", "post", "posted", "further", "popular", "first", "people",
        "had", "have", "has", "may", "might", "will", "can", "cannot", "up",
        "very", "more", "most", "over", "under", "into", "onto", "also",
        "such", "some", "many", "much", "few", "several",
        "who", "what", "when", "where", "why", "how",
        "said", "says", "say", "according", "reported", "reportedly",
        "claimed", "described", "shared", "share", "taken", "took",
        "new", "latest", "week", "month", "year",
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
            re.fullmatch(r"\d+(?:\.\d+)?", cleaned)
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
    digit_tokens = sum(1 for token in tokens if re.fullmatch(r"\d+(?:\.\d+)?", token))
    if digit_tokens / max(len(tokens), 1) > 0.6:
        return True
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return True
    if digit_tokens / max(len(tokens), 1) > 0.6:
        return True
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return True
    return False


def _core_entities(entities: List[str]) -> List[str]:
    generic = {"us", "u s", "u.s.", "uk", "eu"}
    core = [entity for entity in entities if _normalize_for_match(entity) not in generic]
    return core or entities[:1]


def build_query_candidates(claim: str) -> List[str]:
    """Build search queries from *most context* to *least context*.

    Strategy (hybrid):
      1. Cleaned full sentence — preserves semantic structure, best recall
      2. Entity-rich keyword combos — targeted when full sentence is too noisy
      3. Progressively shorter keyword combos — broadest fallback
    """
    candidates: List[str] = []
    seen: set = set()

    def _add(query: str) -> None:
        query = _sanitize_gnews_query(query)
        lowered = query.lower()
        if not query or lowered in seen or _is_bad_query(query):
            return
        seen.add(lowered)
        candidates.append(query)

    # ── Phase 1: cleaned full sentence (best context) ─────────────────────
    sentence_query = _sanitize_gnews_sentence_query(claim)
    if sentence_query and not _is_bad_query(sentence_query):
        _add(sentence_query)

    # ── Phase 2: keyword-based candidates (fallback) ──────────────────────
    keywords = _extract_keywords(claim, max_terms=10)
    entities = _extract_capitalized_entities(claim)
    numbers = _extract_numbers(claim)
    keywords_without_entities = _exclude_entity_terms(keywords, entities)

    non_numeric_keywords = [
        kw for kw in keywords_without_entities
        if not re.fullmatch(r"\d+(?:\.\d+)?", kw)
    ]

    # Entity + number + keywords (long → short)
    long_terms = _dedupe_terms(entities[:4] + numbers[:2] + non_numeric_keywords[:8])
    for size in (10, 8, 6, 4):
        if len(long_terms) >= size:
            _add(" ".join(long_terms[:size]))

    # Entity + keyword combos
    if entities and len(non_numeric_keywords) >= 2:
        _add(" ".join(_dedupe_terms(entities[:1] + non_numeric_keywords[1:3])))
        _add(" ".join(_dedupe_terms(entities[:1] + non_numeric_keywords[-2:])))

    if len(entities) >= 2 and len(non_numeric_keywords) >= 3:
        _add(" ".join(_dedupe_terms(entities[1:2] + non_numeric_keywords[:3])))

    if entities and keywords_without_entities:
        _add(" ".join(_dedupe_terms(entities[:2] + keywords_without_entities[:3])))

    if entities and numbers:
        _add(" ".join(_dedupe_terms(entities[:2] + numbers[:2])))

    if entities and non_numeric_keywords:
        _add(" ".join(_dedupe_terms(entities[:1] + non_numeric_keywords[:2])))

    # Pure keyword combos
    if len(non_numeric_keywords) >= 4:
        _add(" ".join(non_numeric_keywords[:4]))
    if len(non_numeric_keywords) >= 3:
        _add(" ".join(non_numeric_keywords[:3]))
    elif not entities and len(non_numeric_keywords) == 2:
        _add(" ".join(non_numeric_keywords[:2]))

    # Entity-only (broadest)
    if entities:
        _add(" ".join(_dedupe_terms(entities[:2])))

    if keywords:
        _add(" ".join(keywords[:5]))

    return candidates[:MAX_QUERY_CANDIDATES]


def _article_combined_text(article: Dict) -> str:
    title = article.get("title", "") or ""
    description = article.get("description", "") or ""
    content = article.get("content", "") or ""
    return f"{title}. {description}. {content}".strip()


def _weighted_keyword_match(keywords: List[str], text: str) -> float:
    if not keywords:
        return 0.0

    total_weight = 0.0
    hit_weight = 0.0
    seen = set()
    for keyword in keywords:
        for term in _sanitize_gnews_query(keyword).split():
            lowered = term.lower()
            if lowered in seen or lowered in _RERANK_STOPWORDS or len(lowered) <= 2:
                continue
            seen.add(lowered)
            weight = 1.8 if len(lowered) >= 7 else 1.0
            total_weight += weight
            if _term_in_text(term, text):
                hit_weight += weight

    if total_weight == 0:
        return 0.0
    return hit_weight / total_weight


def _format_gnews_articles(articles: List[Dict], claim: str) -> List[Dict]:
    results = []
    claim_entities = _extract_capitalized_entities(claim)
    core_entities = _core_entities(claim_entities)
    claim_numbers = _extract_numbers(claim)
    claim_keywords = _extract_keywords(claim, max_terms=12)
    claim_terms = set(_content_tokens(" ".join(claim_keywords) or claim))

    for idx, article in enumerate(articles):
        source_info = article.get("source", {}) or {}
        title = (article.get("title", "") or "")[:300]
        text = (article.get("content", "") or article.get("description", "") or "")[:3000]
        combined_text = f"{title}. {text}".strip()
        full_text = _article_combined_text(article)

        title_overlap = _token_overlap_ratio(claim, title)
        text_overlap = _token_overlap_ratio(claim, full_text)
        entity_match = _match_ratio(claim_entities, full_text)
        core_entity_match = _match_ratio(core_entities, full_text)
        number_match = _match_ratio(claim_numbers, full_text)
        keyword_match = _weighted_keyword_match(claim_keywords, full_text)
        api_rank_score = 1.0 / float(article.get("_api_rank", idx + 1))

        credibility = _get_source_credibility(source_info.get("name", ""))

        final_score = (
            0.18 * title_overlap
            + 0.18 * text_overlap
            + 0.24 * core_entity_match
            + 0.12 * entity_match
            + 0.18 * keyword_match
            + 0.05 * number_match
            + 0.03 * api_rank_score
            + 0.02 * credibility
        )

        matched_entities = _match_items(claim_entities, full_text)
        matched_numbers = _match_items(claim_numbers, full_text)
        matched_keywords = [
            term
            for term in sorted(claim_terms)
            if _term_in_text(term, full_text)
        ]

        if core_entities and core_entity_match == 0:
            continue
        if claim_entities and entity_match == 0:
            continue
        if claim_numbers and number_match == 0 and final_score < 0.45:
            continue
            
        # For claims without any entities, require much stronger text/keyword overlap
        # to prevent matching completely unrelated articles.
        if not claim_entities and (keyword_match < 0.4 and text_overlap < 0.25):
            continue

        if keyword_match < MIN_KEYWORD_MATCH and text_overlap < 0.20:
            continue
        if final_score < MIN_EVIDENCE_SCORE:
            continue

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
            "core_entity_match_score": round(core_entity_match, 4),
            "keyword_match_score": round(keyword_match, 4),
            "number_match_score": round(number_match, 4),
            "matched_entities": matched_entities,
            "matched_numbers": matched_numbers,
            "matched_keywords": matched_keywords,
            "credibility_score": round(credibility, 4),
            "gnews_query": article.get("_gnews_query", ""),
            "gnews_sortby": article.get("_gnews_sortby", ""),
            "score": round(final_score, 4),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


_MAX_RETRIES = GNEWS_MAX_RETRIES
_RETRY_BASE_DELAY = GNEWS_RETRY_BASE_DELAY


def _collect_gnews_messages(value: Any) -> List[str]:
    messages = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "message" and isinstance(item, str):
                messages.append(item)
            else:
                messages.extend(_collect_gnews_messages(item))
    elif isinstance(value, list):
        for item in value:
            messages.extend(_collect_gnews_messages(item))
    elif isinstance(value, str):
        messages.append(value)
    return messages


def _empty_response_diagnostic(data: Dict) -> str:
    total = data.get("totalArticles")
    messages = []
    if total not in (None, 0, "0"):
        messages.append(f"GNews returned totalArticles={total}, but no article objects.")
    messages.extend(_collect_gnews_messages(data.get("information", {})))
    messages.extend(_collect_gnews_messages(data.get("articlesRemovedFromResponse", {})))
    messages.extend(_collect_gnews_messages(data.get("errors", {})))
    return " ".join(messages) or "GNews returned no articles."


def _search_once(query: str, max_articles: int, sortby: str) -> Dict:
    if GNEWS_DIRECT_SENTENCE_QUERY:
        query = _sanitize_gnews_sentence_query(query)
    else:
        query = _sanitize_gnews_query(query)
    if _is_bad_query(query):
        return {
            "ok": False,
            "articles": [],
            "error": "No valid GNews query after sanitization.",
        }

    params = {
        "q": query,
        "max": max_articles,
        "apikey": GNEWS_API_KEY,
        "lang": "en",
        "sortby": sortby,
        "in": GNEWS_SEARCH_FIELDS,
        "nullable": "description,content",
        "expand": "content",
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
        articles = data.get("articles", [])
        if not articles:
            return {
                "ok": False,
                "articles": [],
                "error": _empty_response_diagnostic(data),
            }
        for idx, article in enumerate(articles, start=1):
            article["_gnews_query"] = query
            article["_gnews_sortby"] = sortby
            article["_api_rank"] = idx
        return {
            "ok": True,
            "articles": articles,
            "error": None,
        }

    return {"ok": False, "articles": [], "error": last_error or "Max retries exceeded."}


def _search_query(query: str, max_articles: int) -> Dict:
    errors = []
    for sortby in GNEWS_SORTBY_ORDER:
        result = _search_once(query, max_articles=max_articles, sortby=sortby)
        if result["ok"]:
            return result
        if result.get("error"):
            errors.append(f"{sortby}: {result['error']}")
    return {
        "ok": False,
        "articles": [],
        "error": " | ".join(errors) if errors else "GNews returned no articles.",
    }


def _article_key(article: Dict) -> str:
    url = str(article.get("url", "")).strip()
    title = str(article.get("title", "")).strip()
    published_at = str(article.get("publishedAt", "")).strip()
    return url if url else f"{title}__{published_at}"


# ── Evidence cache ───────────────────────────────────────────────────────────
# Keyed by (claim_lower, top_k) — avoids re-hitting GNews for repeated claims
_evidence_cache: dict = {}
_CACHE_MAX_SIZE = EVIDENCE_CACHE_MAX_SIZE


def _cache_key(claim: str, top_k: int) -> str:
    import hashlib
    settings = "|".join([
        ",".join(GNEWS_SORTBY_ORDER),
        GNEWS_SEARCH_FIELDS,
        str(GNEWS_RESULTS_PER_QUERY),
        str(MIN_EVIDENCE_SCORE),
        str(MIN_KEYWORD_MATCH),
        str(GNEWS_DIRECT_SENTENCE_QUERY),
        str(MAX_GNEWS_QUERY_CHARS),
    ])
    text = f"{claim.lower().strip()}|{top_k}|{settings}"
    return hashlib.md5(text.encode()).hexdigest()


def clear_evidence_cache() -> None:
    """Call this to invalidate all cached results (e.g. after API key change)."""
    _evidence_cache.clear()


def _retrieve_evidence_legacy(claim: str, top_k: int = TOP_K_EVIDENCE) -> Dict:
    """
    Return format:
    {
        "query": "...",           # all search queries actually used
        "results": [...],         # ranked evidence list
        "error": None or "..."
    }
    """
    # Cache check
    ck = _cache_key(claim, top_k)
    if ck in _evidence_cache:
        return _evidence_cache[ck]

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
    fetch_count = min(max(GNEWS_RESULTS_PER_QUERY, top_k * 4), 100)
    target_article_count = max(EARLY_STOP_ARTICLE_COUNT, top_k * 6)
    seen_keys = set()          # ← 移到循环外，跨 query 累积去重

    for query in query_candidates:
        search_result = _search_query(query, max_articles=fetch_count)
        used_queries.append(query)

        if not search_result["ok"]:
            last_error = search_result["error"]
            continue

        for article in search_result["articles"]:
            key = _article_key(article)
            if key not in seen_keys:
                seen_keys.add(key)
                all_articles.append(article)

        if len(seen_keys) >= target_article_count:
            break

    if not all_articles:
        return {
            "query": " | ".join(query_candidates),
            "results": [],
            "error": last_error,
        }

    ranked_results = _format_gnews_articles(all_articles, claim)

    if not ranked_results:
        result = {
            "query": " | ".join(used_queries),
            "results": [],
            "error": "GNews returned articles, but none passed the local relevance filter.",
        }
        _evidence_cache[ck] = result
        return result

    result = {
        "query": " | ".join(used_queries),
        "results": ranked_results[:top_k],
        "error": None,
    }

    # Write to cache (evict oldest if full)
    if len(_evidence_cache) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_evidence_cache))
        del _evidence_cache[oldest_key]
    _evidence_cache[ck] = result
    return result


def retrieve_evidence(claim: str, top_k: int = TOP_K_EVIDENCE) -> Dict:
    """Search generated queries from longest to shortest until usable evidence appears."""
    ck = _cache_key(claim, top_k)
    if ck in _evidence_cache:
        return _evidence_cache[ck]

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

    used_queries = []
    all_articles = []
    seen_keys = set()
    last_error = None
    ranked_results = []
    fetch_count = min(max(GNEWS_RESULTS_PER_QUERY, top_k * 4), 100)
    target_article_count = max(EARLY_STOP_ARTICLE_COUNT, top_k * 6)

    for query in query_candidates:
        search_result = _search_query(query, max_articles=fetch_count)
        used_queries.append(query)

        if not search_result["ok"]:
            last_error = search_result["error"]
            continue

        for article in search_result["articles"]:
            key = _article_key(article)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_articles.append(article)

        ranked_results = _format_gnews_articles(all_articles, claim)
        if ranked_results and (len(ranked_results) >= top_k or len(seen_keys) >= target_article_count):
            break

    if not all_articles:
        return {
            "query": " | ".join(used_queries),
            "results": [],
            "error": last_error,
        }

    if not ranked_results:
        result = {
            "query": " | ".join(used_queries),
            "results": [],
            "error": (
                "GNews returned articles for the generated queries, "
                "but none passed the local relevance filter."
            ),
        }
    else:
        result = {
            "query": " | ".join(used_queries),
            "results": ranked_results[:top_k],
            "error": None,
        }

    if len(_evidence_cache) >= _CACHE_MAX_SIZE:
        del _evidence_cache[next(iter(_evidence_cache))]
    _evidence_cache[ck] = result
    return result
