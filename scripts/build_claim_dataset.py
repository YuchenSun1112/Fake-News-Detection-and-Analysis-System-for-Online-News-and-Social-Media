import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BBC_FEEDS = {
    "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "health": "https://feeds.bbci.co.uk/news/health/rss.xml",
    "uk": "https://feeds.bbci.co.uk/news/uk/rss.xml",
}

GOOGLE_PRESETS = {"none", "broad"}

DEFAULT_PROMPT = (
    "Summarize the news passage into short retrieval-ready factual statements.\n"
    "Rules:\n"
    "- Keep only verifiable facts.\n"
    "- Preserve named entities, numbers, dates, locations, and outcomes.\n"
    "- Each line must be a complete standalone sentence with an explicit subject.\n"
    "- Prefer concise neutral wording.\n"
    "- Do not output opinions, questions, fragments, or vague pronouns.\n\n"
    "Passage: {passage}\n\n"
    "Retrieval summaries:\n"
)


def _fetch_url(url: str, timeout: int = 20) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; claim-dataset-builder/1.0; "
                "+https://example.local)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _strip_html(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_child_text(element: ET.Element, local_name: str) -> str:
    for child in list(element):
        if child.tag.split("}")[-1] == local_name:
            return child.text or ""
    return ""


def _parse_rss_items(feed_xml: bytes, source_name: str) -> List[Dict]:
    root = ET.fromstring(feed_xml)
    items = []

    for item in root.findall(".//item"):
        title = _strip_html(item.findtext("title", default=""))
        description = _strip_html(item.findtext("description", default=""))
        link = _strip_html(item.findtext("link", default=""))
        published_at = _strip_html(item.findtext("pubDate", default=""))
        publisher = _strip_html(_find_child_text(item, "source")) or source_name

        if not title:
            continue

        passage = f"{title}. {description}".strip()
        items.append(
            {
                "source": source_name,
                "publisher": publisher,
                "title": title,
                "description": description,
                "url": link,
                "published_at": published_at,
                "passage": passage,
            }
        )

    return items


def fetch_bbc_items(categories: Iterable[str], pause_seconds: float) -> List[Dict]:
    all_items = []
    for category in categories:
        feed_url = BBC_FEEDS.get(category)
        if not feed_url:
            raise ValueError(f"Unknown BBC category: {category}")
        try:
            feed_xml = _fetch_url(feed_url)
            all_items.extend(_parse_rss_items(feed_xml, "BBC News"))
        except Exception as exc:
            print(f"Warning: failed to fetch BBC category '{category}': {exc}")
        time.sleep(pause_seconds)
    return all_items


def _google_news_url(query: str) -> str:
    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(params)


def fetch_google_news_items(
    queries: Iterable[str],
    pause_seconds: float,
    max_raw_items: int = None,
) -> List[Dict]:
    all_items = []
    for query in queries:
        if max_raw_items is not None and len(all_items) >= max_raw_items:
            break
        try:
            feed_xml = _fetch_url(_google_news_url(query))
            items = _parse_rss_items(feed_xml, "Google News")
        except Exception as exc:
            print(f"Warning: failed to fetch Google News query '{query}': {exc}")
            time.sleep(pause_seconds)
            continue
        for item in items:
            item["search_query"] = query
        all_items.extend(items)
        time.sleep(pause_seconds)
    return all_items


def build_google_preset_queries(preset: str) -> List[str]:
    if preset == "none":
        return []
    if preset != "broad":
        raise ValueError(f"Unknown Google News preset: {preset}")

    regions = [
        "United States",
        "United Kingdom",
        "Europe",
        "China",
        "India",
        "Japan",
        "Middle East",
        "Africa",
        "Latin America",
        "Ukraine",
        "Russia",
        "Israel",
        "Gaza",
        "Iran",
        "Taiwan",
        "South Korea",
        "France",
        "Germany",
        "Canada",
        "Australia",
    ]
    topics = [
        "politics",
        "election",
        "government",
        "court",
        "economy",
        "inflation",
        "interest rates",
        "stock market",
        "business",
        "trade",
        "technology",
        "artificial intelligence",
        "cybersecurity",
        "space",
        "science",
        "health",
        "medicine",
        "climate change",
        "energy",
        "education",
        "immigration",
        "crime",
        "transportation",
        "sports",
        "entertainment",
    ]
    event_queries = [
        "breaking news",
        "latest world news",
        "latest business news",
        "latest technology news",
        "latest health news",
        "latest science news",
        "latest climate news",
        "latest politics news",
        "international news",
        "global economy news",
        "central bank news",
        "company earnings news",
        "supply chain news",
        "oil prices news",
        "renewable energy news",
        "electric vehicle news",
        "semiconductor news",
        "social media regulation news",
        "data privacy news",
        "public health news",
        "vaccine news",
        "extreme weather news",
        "natural disaster news",
        "military news",
        "diplomacy news",
        "human rights news",
        "labor market news",
        "housing market news",
        "aviation news",
        "shipping news",
    ]

    queries = list(event_queries)
    for region in regions:
        for topic in topics:
            queries.append(f"{region} {topic} news")

    deduped = []
    seen = set()
    for query in queries:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(query)
    return deduped


def read_query_file(path: Path) -> List[str]:
    if not path:
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _stable_id(prefix: str, item: Dict) -> str:
    raw = item.get("url") or f"{item.get('title', '')}|{item.get('published_at', '')}"
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    safe_prefix = re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")
    return f"{safe_prefix}_{digest}"


def _dedupe_items(items: Iterable[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for item in items:
        key = item.get("url") or f"{item.get('title')}|{item.get('published_at')}"
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _round_robin_batches(batches: List[List[Dict]]) -> List[Dict]:
    mixed = []
    max_len = max((len(batch) for batch in batches), default=0)
    for idx in range(max_len):
        for batch in batches:
            if idx < len(batch):
                mixed.append(batch[idx])
    return mixed


def build_training_records(
    news_items: Iterable[Dict],
    top_k_claims: int,
    min_claims: int,
    min_passage_chars: int,
) -> List[Dict]:
    from src.claim_extractor import extract_claims
    from src.cleaner import clean_article_text

    records = []
    for item in news_items:
        passage = clean_article_text(item.get("passage", ""))
        if len(passage) < min_passage_chars:
            continue

        claims = extract_claims(passage, top_k=top_k_claims)
        claim_texts = [claim["text"] for claim in claims if claim.get("text")]

        if len(claim_texts) < min_claims:
            continue

        record = {
            "id": _stable_id(item.get("source", "news"), item),
            "source": item.get("source", ""),
            "publisher": item.get("publisher", ""),
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "published_at": item.get("published_at", ""),
            "search_query": item.get("search_query", ""),
            "passage": passage,
            "input_text": DEFAULT_PROMPT.format(passage=passage),
            "claims": claims,
            "target_text": "\n".join(claim_texts),
        }
        records.append(record)

    return records


def write_dataset(records: List[Dict], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a claim extraction training dataset from BBC RSS and/or "
            "Google News RSS snippets."
        )
    )
    parser.add_argument(
        "--source",
        choices=["bbc", "google", "all"],
        default="all",
        help="News source to fetch.",
    )
    parser.add_argument(
        "--bbc-category",
        action="append",
        choices=sorted(BBC_FEEDS.keys()),
        help="BBC RSS category. Can be provided multiple times.",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Google News search query. Can be provided multiple times.",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Text file with one Google News search query per line.",
    )
    parser.add_argument(
        "--google-preset",
        choices=sorted(GOOGLE_PRESETS),
        default="none",
        help="Add a built-in Google News query set.",
    )
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument(
        "--target-records",
        type=int,
        default=None,
        help="Keep at most this many successfully labeled training records.",
    )
    parser.add_argument("--top-k-claims", type=int, default=5)
    parser.add_argument("--min-claims", type=int, default=1)
    parser.add_argument("--min-passage-chars", type=int, default=40)
    parser.add_argument(
        "--extractor-mode",
        choices=["env", "rule", "model", "summary", "summary_model"],
        default="rule",
        help=(
            "Claim extractor mode. Use 'rule' for fast dataset bootstrapping, "
            "or 'model'/'summary_model' if the seq2seq model is already available."
        ),
    )
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing an empty dataset. By default, empty results abort.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "claim_extraction_dataset.json",
    )
    parser.add_argument("--format", choices=["json", "jsonl"], default="json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.extractor_mode != "env":
        os.environ["CLAIM_EXTRACTOR_MODE"] = args.extractor_mode

    bbc_categories = args.bbc_category or ["world", "business", "technology"]
    google_queries = []
    google_queries.extend(build_google_preset_queries(args.google_preset))
    google_queries.extend(read_query_file(args.query_file) if args.query_file else [])
    google_queries.extend(args.query or [])
    if not google_queries:
        google_queries = ["latest world news", "technology news", "business news"]

    news_batches = []
    if args.source in {"bbc", "all"}:
        news_batches.append(fetch_bbc_items(bbc_categories, args.pause_seconds))
    if args.source in {"google", "all"}:
        news_batches.append(
            fetch_google_news_items(
                google_queries,
                args.pause_seconds,
                max_raw_items=max(args.max_items * 2, args.max_items + 100),
            )
        )

    news_items = _dedupe_items(_round_robin_batches(news_batches))[: args.max_items]
    records = build_training_records(
        news_items=news_items,
        top_k_claims=args.top_k_claims,
        min_claims=args.min_claims,
        min_passage_chars=args.min_passage_chars,
    )
    if args.target_records is not None:
        records = records[: args.target_records]

    if not records and not args.allow_empty:
        raise SystemExit("No training records were generated; output file was not modified.")

    write_dataset(records, args.output, args.format)
    print(f"Fetched news items: {len(news_items)}")
    print(f"Training records written: {len(records)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
