import re
import time
import requests
from typing import List, Dict

from config import GNEWS_API_KEY, GNEWS_ENDPOINT, TOP_K_EVIDENCE, REQUEST_TIMEOUT


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def _token_overlap_ratio(a: str, b: str) -> float:
    """
    计算 claim 和 evidence 的词重合度
    """
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _extract_keywords(claim: str, max_terms: int = 8) -> List[str]:
    """
    从 claim 中提取更适合搜索的关键词
    """
    raw_words = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9\-]*\b", str(claim))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
        "that", "this", "these", "those", "it", "its", "he", "she", "they",
        "his", "her", "their", "and", "or", "but", "if", "then", "than",
        "just", "soon", "could", "would", "should", "after", "before",
        "said", "says", "say"
    }

    keywords = []
    for w in raw_words:
        wl = w.lower()
        if wl in stopwords:
            continue

        # 保留数字、首字母大写词、较长词
        if w.isdigit() or w[:1].isupper() or len(w) >= 5:
            keywords.append(w)

    # 去重并保持顺序
    seen = set()
    deduped = []
    for w in keywords:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            deduped.append(w)

    return deduped[:max_terms]


def build_search_query(claim: str, max_len: int = 180) -> str:
    """
    把原始 claim 转成更适合 GNews 搜索的 query
    """
    claim = str(claim).strip()

    keywords = _extract_keywords(claim, max_terms=8)

    if keywords:
        query = " ".join(keywords)
    else:
        # 如果抽不出关键词，就退回到清洗后的原句
        query = re.sub(r'[^\w\s"]', " ", claim)
        query = re.sub(r"\s+", " ", query).strip()

    # 防止过长
    if len(query) > max_len:
        query = query[:max_len].rsplit(" ", 1)[0]

    return query


def _format_gnews_articles(articles: List[Dict], claim: str) -> List[Dict]:
    """
    整理 GNews 返回结果，并做一个简单本地重排
    """
    results = []

    for idx, article in enumerate(articles):
        source_info = article.get("source", {}) or {}
        title = article.get("title", "") or ""
        text = article.get("description", "") or article.get("content", "") or ""
        combined = f"{title} {text}".strip()

        overlap = _token_overlap_ratio(claim, combined)

        results.append(
            {
                "evidence_id": idx + 1,
                "title": title,
                "text": text,
                "source": source_info.get("name", "Unknown"),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "api_rank_score": round(1.0 / (idx + 1), 4),
                "local_match_score": round(overlap, 4),
                "score": round((1.0 / (idx + 1)) * 0.3 + overlap * 0.7, 4),
            }
        )

    # 本地按综合分数重排
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def retrieve_evidence(claim: str, top_k: int = TOP_K_EVIDENCE) -> Dict:
    """
    返回格式：
    {
        "query": "...",
        "results": [...]
    }
    """
    if not GNEWS_API_KEY:
        raise ValueError("GNEWS_API_KEY is missing. Please set it in config.py or environment.")

    query = build_search_query(claim)

    params = {
        "q": query,
        "max": top_k,
        "apikey": GNEWS_API_KEY,
        "lang": "en",
        "sortby": "relevance",
        "in": "title,description",
    }

    response = requests.get(GNEWS_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)

    # 免费版 GNews 限速比较严格，避免多个 claim 连续请求过快
    time.sleep(1.05)

    if response.status_code != 200:
        try:
            error_payload = response.json()
        except Exception:
            error_payload = response.text

        return {
            "query": query,
            "results": [],
            "error": f"GNews API error {response.status_code}: {error_payload}",
        }

    data = response.json()
    articles = data.get("articles", [])

    formatted_results = _format_gnews_articles(articles, claim)

    return {
        "query": query,
        "results": formatted_results,
        "error": None,
    }