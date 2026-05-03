# Fact-based News Verification System

A Streamlit-based prototype for news verification that combines:

- a **baseline fake/real classifier**
- a **claim extraction + evidence retrieval + claim verification pipeline**

The goal is to move beyond style-based fake news classification and toward **evidence-based verification**.

## Overview

Given a news article, the system:

1. cleans the input text
2. runs a baseline BERT classifier
3. extracts key factual claims
4. retrieves related evidence from GNews
5. labels each claim as `supported`, `refuted`, or `not enough information`
6. aggregates the claim-level results into a final verdict

Possible final verdicts:

- `Likely True`
- `Likely False`
- `Unverified`

## Repository Structure

```text
.
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── gossipcop_fake.csv
│   ├── gossipcop_real.csv
│   ├── politifact_fake.csv
│   └── politifact_real.csv
└── src/
    ├── aggregator.py
    ├── baseline_model.py
    ├── claim_extractor.py
    ├── cleaner.py
    ├── data_loader.py
    ├── retriever.py
    └── verifier.py
```

## File Guide

### Root Files

- `app.py`: Streamlit app entry point and pipeline controller.
- `config.py`: project paths, API settings, and shared constants.
- `requirements.txt`: Python dependencies.
- `README.md`: project documentation.

### `src/`

- `src/cleaner.py`: text cleaning utilities.
- `src/data_loader.py`: dataset loading and normalization for the baseline model.
- `src/baseline_model.py`: BERT baseline training and inference.
- `src/claim_extractor.py`: claim extraction and ranking logic.
- `src/retriever.py`: GNews query building, retrieval, deduplication, and reranking.
- `src/verifier.py`: heuristic claim verification against retrieved evidence.
- `src/aggregator.py`: final article-level verdict aggregation.

### `data/`

- `data/gossipcop_fake.csv`: fake samples from GossipCop.
- `data/gossipcop_real.csv`: real samples from GossipCop.
- `data/politifact_fake.csv`: fake samples from PolitiFact.
- `data/politifact_real.csv`: real samples from PolitiFact.

Note: in the current repository, these datasets mainly provide titles and metadata rather than full article bodies, so the baseline classifier is effectively closer to a **headline classifier**.

## Installation

```bash
pip install -r requirements.txt
pip install requests sentencepiece
```

## Environment Variables

Create a `.env` file in the project root:

```env
GNEWS_API_KEY=your_gnews_api_key_here
```

## Run

```bash
streamlit run app.py
```


## Current Limitations

- The verifier is heuristic, not a trained NLI model.
- Evidence retrieval depends on the GNews API.
- Verification is based on retrieved snippets, not full article content.

## Summary

This project is an **explainable news verification prototype** that combines classification, claim extraction, retrieval, and verification into a single pipeline.
