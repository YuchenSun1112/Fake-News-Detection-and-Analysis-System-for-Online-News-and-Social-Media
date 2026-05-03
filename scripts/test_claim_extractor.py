import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TEXT = (
    "NASA said its new climate satellite was launched from California on Tuesday. "
    "The agency said the mission will track ocean temperatures and atmospheric conditions. "
    "Officials said the satellite cost about $1.2 billion."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test the configured claim extractor model.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--model-path",
        default="models/claim_extractor/final",
        help="Local fine-tuned model path used when CLAIM_EXTRACTOR_MODEL is not already set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CLAIM_EXTRACTOR_MODE", "model")
    os.environ.setdefault("CLAIM_EXTRACTOR_MODEL", args.model_path)

    from src.claim_extractor import extract_claims

    claims = extract_claims(args.text, top_k=args.top_k)
    print(f"CLAIM_EXTRACTOR_MODE={os.environ.get('CLAIM_EXTRACTOR_MODE')}")
    print(f"CLAIM_EXTRACTOR_MODEL={os.environ.get('CLAIM_EXTRACTOR_MODEL')}")
    print(f"Claims extracted: {len(claims)}")
    for claim in claims:
        print(
            f"{claim.get('claim_id')}. [{claim.get('method')}] "
            f"{claim.get('text')} (score={claim.get('score')})"
        )


if __name__ == "__main__":
    main()
