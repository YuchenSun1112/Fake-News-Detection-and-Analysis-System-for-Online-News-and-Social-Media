# Fact-based News Verification System

A Streamlit prototype for explainable news verification. The app cleans an input
article, runs a baseline fake/real classifier, retrieves evidence from GNews,
uses an NLI verifier for claim-level labels, and aggregates the results into an
article verdict.

## Environment

This project uses `uv` only. Dependencies and default runtime configuration live
in `pyproject.toml`.

```powershell
uv sync
```

If your current virtualenv is already synced and the network is unavailable, use
`uv run --no-sync ...` for local runs. On this Windows setup, point temporary
files at the project-local `.tmp` directory before running heavy torch commands:

```powershell
$env:TEMP=(Resolve-Path .tmp).Path
$env:TMP=$env:TEMP
uv run --no-sync streamlit run app.py
```

The project locks CUDA PyTorch through the PyTorch CUDA 12.8 wheel index:

```toml
torch = { index = "pytorch-cu128" }
```

Check GPU visibility:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Configuration

Default non-secret configuration is in:

```toml
[tool.fact_news]
[tool.fact_news.claim_extractor]
[tool.fact_news.retriever]
[tool.fact_news.nli]
```

`config.py` reads those values and keeps environment variables as temporary
overrides. Keep secrets in `.env`:

```env
GNEWS_API_KEY=your_gnews_api_key_here
```

Do not commit `.env`.

## Run

```powershell
uv run streamlit run app.py
```

If `uv run` tries to redownload packages, use:

```powershell
uv run --no-sync streamlit run app.py
```

## Useful Commands

Build real NLI data from VitaminC:

```powershell
uv run python scripts/build_real_nli_dataset.py --csv
```

Train the NLI verifier:

```powershell
uv run python scripts/train_nli_verifier.py --epochs 2 --train-batch-size 8 --fp16
```

Build weak claim-extraction data:

```powershell
uv run python scripts/build_claim_dataset.py --source google --google-preset broad --max-items 18000 --target-records 5000 --output data/claim_extraction_dataset.json
```

Train the claim extractor:

```powershell
uv run python scripts/train_claim_extractor.py --dataset data/claim_extraction_dataset.json --model-name google/flan-t5-base --output-dir models/claim_extractor/final --epochs 3 --train-batch-size 2 --eval-batch-size 2 --gradient-accumulation-steps 8
```

Test GNews API access:

```powershell
uv run python scripts/test_gnews_api.py
```

## Project Layout

```text
app.py
config.py                  # reads pyproject.toml defaults
pyproject.toml             # dependencies and default config
uv.lock
data/
models/
scripts/
src/
```

## Notes

- `requirements.txt` is intentionally removed. Use `uv sync`.
- `.tmp/` is the project-local temporary directory for uv, torch, and leftover
  blat-style temp files. It is ignored by Git.
- `data/nli_real/` is the real NLI training dataset generated from VitaminC.
- `data/nli/` is the earlier silver dataset and should only be used for quick
  pipeline checks.
- Large model files are expected under `models/` and may require Git LFS.
