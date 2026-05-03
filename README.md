# Fact-based News Verification System

A Streamlit prototype for explainable news verification. The system combines a
baseline fake/real classifier with a claim-level verification pipeline:

1. clean the input article
2. run a baseline BERT fake/real classifier
3. extract factual claims with a fine-tuned FLAN-T5 claim extractor and rule fallback
4. retrieve evidence from GNews
5. label each claim as `supported`, `refuted`, or `not enough information`
6. aggregate claim-level labels into an article-level verdict

Final verdicts are:

- `Likely True`
- `Likely False`
- `Unverified`

## Project Structure

```text
.
|-- app.py
|-- config.py
|-- pyproject.toml
|-- README.md
|-- data/
|   |-- claim_extraction_dataset.json
|   |-- gossipcop_fake.csv
|   |-- gossipcop_real.csv
|   |-- politifact_fake.csv
|   `-- politifact_real.csv
|-- models/
|   |-- baseline_classifier/
|   |   `-- final/
|   `-- claim_extractor/
|       `-- final/
|-- scripts/
|   |-- build_claim_dataset.py
|   |-- test_claim_extractor.py
|   `-- train_claim_extractor.py
`-- src/
    |-- aggregator.py
    |-- baseline_model.py
    |-- claim_extractor.py
    |-- cleaner.py
    |-- data_loader.py
    |-- retriever.py
    `-- verifier.py
```

## Models

This project uses two local model directories:

- `models/baseline_classifier/final`: BERT fake/real classifier
- `models/claim_extractor/final`: fine-tuned FLAN-T5 claim extractor

Large model files are tracked with Git LFS. After cloning the repository, make
sure Git LFS is installed and model files are pulled:

```bash
git lfs install
git lfs pull
```

If the model files are missing, the app may still run partially, but baseline
classification or model-based claim extraction will be unavailable.

## Environment Setup

This project uses `uv` for environment and dependency management.

Install dependencies:

```bash
uv sync
```

The project targets Python `>=3.10,<3.13`, which is recommended for PyTorch and
Transformers compatibility.

## Environment Variables

Create a `.env` file in the project root. Do not commit this file.

```env
GNEWS_API_KEY=your_gnews_api_key_here
CLAIM_EXTRACTOR_MODE=model
CLAIM_EXTRACTOR_MODEL=models/claim_extractor/final
```

Notes:

- `GNEWS_API_KEY` is required for evidence retrieval.
- `CLAIM_EXTRACTOR_MODE=model` makes the app try the trained claim extractor first.
- If the model fails to load or produces no usable claims, `src/claim_extractor.py`
  falls back to rule-based extraction.

## Run the App

```bash
uv run streamlit run app.py
```

Paste a news article into the text box and click `Verify News`.

## Utility Scripts

Test the configured claim extractor:

```bash
uv run python scripts/test_claim_extractor.py
```

Build a weakly labeled claim extraction dataset from Google News RSS:

```bash
uv run python scripts/build_claim_dataset.py --source google --google-preset broad --max-items 18000 --target-records 5000 --output data/claim_extraction_dataset.json
```

Train the claim extractor:

```bash
uv run python scripts/train_claim_extractor.py --dataset data/claim_extraction_dataset.json --model-name google/flan-t5-base --output-dir models/claim_extractor/final --epochs 3 --train-batch-size 2 --eval-batch-size 2 --gradient-accumulation-steps 8
```

For a quick smoke test:

```bash
uv run python scripts/train_claim_extractor.py --debug-sample-size 200 --epochs 1 --output-dir models/claim_extractor/debug
```

## Baseline Classifier

The baseline classifier is trained on the included GossipCop and PolitiFact CSV
files. These files mainly contain titles and metadata rather than full article
bodies, so the baseline model is closer to a headline-level fake/real classifier.

The baseline result is shown as a secondary signal. The main explainable pipeline
comes from claim extraction, evidence retrieval, claim verification, and
aggregation.

## Current Limitations

- Claim extraction is trained from weak labels generated from news RSS snippets.
- Evidence retrieval depends on GNews availability and API limits.
- Claim verification is heuristic and relies on lexical overlap, entity matches,
  numeric consistency, and negation signals.
- Retrieved evidence uses article titles/descriptions rather than full article text.
- The final verdict is a coarse aggregation of claim-level labels.

## Git LFS Notes

The repository is configured to track large model binaries with Git LFS:

```text
*.safetensors
*.bin
*.pt
*.pth
```

Before pushing model files, check:

```bash
git lfs ls-files
```

Do not commit `.env` because it contains private API keys.
