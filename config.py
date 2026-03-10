import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

GOSSIPCOP_FAKE = os.path.join(DATA_DIR, "gossipcop_fake.csv")
GOSSIPCOP_REAL = os.path.join(DATA_DIR, "gossipcop_real.csv")
POLITIFACT_FAKE = os.path.join(DATA_DIR, "politifact_fake.csv")
POLITIFACT_REAL = os.path.join(DATA_DIR, "politifact_real.csv")
EVIDENCE_CORPUS_PATH = os.path.join(DATA_DIR, "evidence_corpus.csv")

BASELINE_MODEL_DIR = os.path.join(MODELS_DIR, "baseline_classifier")
CLAIM_VERIFIER_DIR = os.path.join(MODELS_DIR, "claim_verifier")

BASELINE_MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 256
TOP_K_CLAIMS = 5
TOP_K_EVIDENCE = 5
REQUEST_TIMEOUT = 15

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "").strip()
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"

REFUTED_WEIGHT = 1.2
SUPPORTED_WEIGHT = 1.0
NEI_WEIGHT = 0.4