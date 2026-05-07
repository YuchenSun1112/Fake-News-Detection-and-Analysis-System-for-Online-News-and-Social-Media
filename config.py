import os
from pathlib import Path

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
PYPROJECT_PATH = BASE_DIR / "pyproject.toml"

load_dotenv(dotenv_path=ENV_PATH)


def _tool_config() -> dict:
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    return data.get("tool", {}).get("fact_news", {})


_CFG = _tool_config()


def _section(name: str) -> dict:
    return _CFG.get(name, {})


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, str(default)).strip()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


DATA_DIR = str(BASE_DIR / "data")
MODELS_DIR = str(BASE_DIR / "models")

GOSSIPCOP_FAKE = str(Path(DATA_DIR) / "gossipcop_fake.csv")
GOSSIPCOP_REAL = str(Path(DATA_DIR) / "gossipcop_real.csv")
POLITIFACT_FAKE = str(Path(DATA_DIR) / "politifact_fake.csv")
POLITIFACT_REAL = str(Path(DATA_DIR) / "politifact_real.csv")
EVIDENCE_CORPUS_PATH = str(Path(DATA_DIR) / "evidence_corpus.csv")

BASELINE_MODEL_DIR = str(Path(MODELS_DIR) / "baseline_classifier")
CLAIM_VERIFIER_DIR = str(Path(MODELS_DIR) / "claim_verifier")

BASELINE_MODEL_NAME = _env_str("BASELINE_MODEL_NAME", _CFG["baseline_model_name"])
MAX_LENGTH = _env_int("MAX_LENGTH", _CFG["max_length"])
TOP_K_CLAIMS = _env_int("TOP_K_CLAIMS", _CFG["top_k_claims"])
TOP_K_EVIDENCE = _env_int("TOP_K_EVIDENCE", _CFG["top_k_evidence"])
REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", _CFG["request_timeout"])

GNEWS_API_KEY = _env_str("GNEWS_API_KEY", "")
GNEWS_ENDPOINT = _env_str("GNEWS_ENDPOINT", _CFG["gnews_endpoint"])

SUPPORTED_WEIGHT = _env_float("SUPPORTED_WEIGHT", _CFG["supported_weight"])
REFUTED_WEIGHT = _env_float("REFUTED_WEIGHT", _CFG["refuted_weight"])
NEI_WEIGHT = _env_float("NEI_WEIGHT", _CFG["nei_weight"])
BASELINE_PRIOR_WEIGHT = _env_float("BASELINE_PRIOR_WEIGHT", _CFG["baseline_prior_weight"])

_CLAIM = _section("claim_extractor")
CLAIM_EXTRACTOR_MODE = _env_str("CLAIM_EXTRACTOR_MODE", _CLAIM["mode"])
CLAIM_EXTRACTOR_MODEL = _env_str("CLAIM_EXTRACTOR_MODEL", _CLAIM["model"])
MIN_SENT_WORDS = _env_int("MIN_SENT_WORDS", _CLAIM["min_sent_words"])
MIN_CLAIM_WORDS = _env_int("MIN_CLAIM_WORDS", _CLAIM["min_claim_words"])
MAX_CLAIM_WORDS = _env_int("MAX_CLAIM_WORDS", _CLAIM["max_claim_words"])
MAX_INPUT_CHARS_PER_SENT = _env_int(
    "MAX_INPUT_CHARS_PER_SENT",
    _CLAIM["max_input_chars_per_sent"],
)
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", _CLAIM["max_new_tokens"])
MAX_INPUT_CHARS_PER_SUMMARY_CHUNK = _env_int(
    "MAX_INPUT_CHARS_PER_SUMMARY_CHUNK",
    _CLAIM["max_input_chars_per_summary_chunk"],
)
MAX_SUMMARY_CHUNKS = _env_int("MAX_SUMMARY_CHUNKS", _CLAIM["max_summary_chunks"])
MIN_SUMMARY_CHUNK_CHARS = _env_int(
    "MIN_SUMMARY_CHUNK_CHARS",
    _CLAIM["min_summary_chunk_chars"],
)
MAX_RETRIEVAL_SUMMARIES_PER_CHUNK = _env_int(
    "MAX_RETRIEVAL_SUMMARIES_PER_CHUNK",
    _CLAIM["max_retrieval_summaries_per_chunk"],
)
MAX_CLAIMS_PER_SENTENCE = _env_int(
    "MAX_CLAIMS_PER_SENTENCE",
    _CLAIM["max_claims_per_sentence"],
)
CLAIM_DEDUP_SIM_THRESHOLD = _env_float(
    "CLAIM_DEDUP_SIM_THRESHOLD",
    _CLAIM["dedup_sim_threshold"],
)

_RETRIEVER = _section("retriever")
MAX_QUERY_CANDIDATES = _env_int("MAX_QUERY_CANDIDATES", _RETRIEVER["max_query_candidates"])
EARLY_STOP_ARTICLE_COUNT = _env_int(
    "EARLY_STOP_ARTICLE_COUNT",
    _RETRIEVER["early_stop_article_count"],
)
MAX_GNEWS_QUERY_WORDS = _env_int("MAX_GNEWS_QUERY_WORDS", _RETRIEVER["max_gnews_query_words"])
MAX_GNEWS_QUERY_CHARS = _env_int("MAX_GNEWS_QUERY_CHARS", _RETRIEVER["max_gnews_query_chars"])
GNEWS_DIRECT_SENTENCE_QUERY = _env_bool(
    "GNEWS_DIRECT_SENTENCE_QUERY",
    _RETRIEVER["gnews_direct_sentence_query"],
)
GNEWS_SORTBY_ORDER = _env_list("GNEWS_SORTBY_ORDER", _RETRIEVER["gnews_sortby_order"])
GNEWS_SEARCH_FIELDS = _env_str("GNEWS_SEARCH_FIELDS", _RETRIEVER["gnews_search_fields"])
GNEWS_RESULTS_PER_QUERY = _env_int(
    "GNEWS_RESULTS_PER_QUERY",
    _RETRIEVER["gnews_results_per_query"],
)
MIN_EVIDENCE_SCORE = _env_float("MIN_EVIDENCE_SCORE", _RETRIEVER["min_evidence_score"])
MIN_KEYWORD_MATCH = _env_float("MIN_KEYWORD_MATCH", _RETRIEVER["min_keyword_match"])
GNEWS_MAX_RETRIES = _env_int("GNEWS_MAX_RETRIES", _RETRIEVER["gnews_max_retries"])
GNEWS_RETRY_BASE_DELAY = _env_float(
    "GNEWS_RETRY_BASE_DELAY",
    _RETRIEVER["gnews_retry_base_delay"],
)
EVIDENCE_CACHE_MAX_SIZE = _env_int(
    "EVIDENCE_CACHE_MAX_SIZE",
    _RETRIEVER["evidence_cache_max_size"],
)

_NLI = _section("nli")
NLI_MODEL_NAME = _env_str("NLI_MODEL_NAME", _NLI["model"])
NLI_MAX_LENGTH = _env_int("NLI_MAX_LENGTH", _NLI["max_length"])
NLI_MAX_EVIDENCE = _env_int("NLI_MAX_EVIDENCE", _NLI["max_evidence"])
NLI_DECISION_THRESHOLD = _env_float("NLI_DECISION_THRESHOLD", _NLI["decision_threshold"])
NLI_NEI_MARGIN = _env_float("NLI_NEI_MARGIN", _NLI["nei_margin"])
