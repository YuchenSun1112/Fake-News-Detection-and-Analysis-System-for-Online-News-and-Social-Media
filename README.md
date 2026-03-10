# Fact-based News Verification System

A fact-based news verification project built with Python, Streamlit, BERT, and online news retrieval APIs.

This project upgrades a traditional fake-news classifier into a more explainable **fact verification pipeline**.  
Instead of only predicting whether a news article looks fake or real, the system:

1. accepts a news article from the user,
2. extracts key factual claims,
3. searches online news sources for relevant evidence,
4. verifies whether the evidence supports or refutes each claim,
5. aggregates the claim-level results into a final article-level verdict.

---

# 1. Project Overview

## 1.1 Goal

The goal of this project is to determine whether a news article is credible by combining:

- a **baseline fake/real classifier**,
- **claim extraction**,
- **online evidence retrieval via API**,
- **claim-level verification**,
- **article-level aggregation**.

This design makes the system more explainable than a standard text classifier.

---

## 1.2 Core Idea

Traditional fake-news classifiers often learn writing style patterns instead of verifying facts.  
This project improves that by introducing an evidence-based verification pipeline.

### Baseline Classifier
A BERT model is trained on labeled fake/real news data.  
It provides a quick initial prediction.

### Fact Verification Pipeline
The article is then processed in multiple steps:

- extract verifiable claims,
- search the web for supporting or contradictory news reports,
- compare each claim against retrieved evidence,
- output:
  - **Supported**
  - **Refuted**
  - **Not Enough Information**

Finally, the system aggregates the claim-level results into a final verdict such as:

- **Likely True**
- **Likely False**
- **Unverified**

---

# 2. Project Structure

```text
project/
│
├─ app.py
├─ config.py
├─ requirements.txt
├─ README.md
│
├─ data/
│  ├─ gossipcop_fake.csv
│  ├─ gossipcop_real.csv
│  ├─ politifact_fake.csv
│  ├─ politifact_real.csv
│
├─ models/
│  ├─ baseline_classifier/
│  └─ claim_verifier/
│
└─ src/
   ├─ cleaner.py
   ├─ data_loader.py
   ├─ baseline_model.py
   ├─ claim_extractor.py
   ├─ retriever.py
   ├─ verifier.py
   └─ aggregator.py