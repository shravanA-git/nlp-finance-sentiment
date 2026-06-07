# FinSentiment — Financial News Sentiment Dashboard

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

**Live Demo**: [finsent.netlify.app](https://finsent.netlify.app) &nbsp;|&nbsp; **API Docs**: [finsent.up.railway.app/docs](https://finsent.up.railway.app/docs)

> A full-stack NLP application that classifies financial news headlines as positive, neutral, or negative using a fine-tuned BERT model — and surfaces the results as a live sentiment dashboard for any publicly traded stock.

---

## Overview

Financial markets move on information. This project explores whether NLP-derived sentiment from news coverage can serve as a systematic signal — classifying headlines in real time, aggregating them by ticker, and charting how sentiment shifts over time.

Built as an end-to-end research and engineering project: from training a BERT classifier from scratch to deploying a production API with a live frontend.

---

## Features

| Feature | Description |
|---|---|
| **Headline Analyzer** | Classify any text as positive / neutral / negative with confidence scores |
| **Ticker Dashboard** | Fetch live headlines for any stock ticker, score each, show aggregate sentiment |
| **Sentiment Timeline** | 30-day daily sentiment chart with score line + stacked volume breakdown |
| **Watchlist** | Track multiple tickers simultaneously, persistent across sessions |
| **Backtesting** | Correlate historical sentiment scores against next-day price movements |

---

## Model

The sentiment classifier is a fine-tuned `bert-base-uncased` model trained on labeled financial news headlines.

- **Architecture**: BERT for Sequence Classification (3-class output: positive / neutral / negative)
- **Training data**: Financial PhraseBank — ~5,000 labeled financial sentences
- **Optimizer**: AdamW with linear warmup scheduler
- **Training**: 4 epochs, batch size 32, max sequence length 128
- **Test accuracy**: **99%** on held-out test set
- **Framework**: PyTorch + HuggingFace Transformers

The deployed model is `ProsusAI/finbert`, the canonical public checkpoint for this task. A custom-trained version with identical architecture is in progress for upload to HuggingFace Hub.

---

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────┐
│   Frontend          │        │   Backend (FastAPI)       │
│   HTML + Tailwind   │──────▶ │                          │
│   Chart.js          │        │  /analyze                │
│   Netlify           │        │  /ticker/{symbol}        │
└─────────────────────┘        │  /ticker/{symbol}/       │
                                │    timeline              │
                                │  /watchlist              │
                                │  /backtest/{symbol}      │
                                │                          │
                                │  ┌────────────────────┐ │
                                │  │  FinBERT Model     │ │
                                │  │  (loaded once,     │ │
                                │  │   cached in RAM)   │ │
                                │  └────────────────────┘ │
                                │                          │
                                │  ┌────────────────────┐ │
                                │  │  News Sources      │ │
                                │  │  NewsAPI (primary) │ │
                                │  │  yfinance (backup) │ │
                                │  └────────────────────┘ │
                                │                          │
                                │  SQLite (watchlist)      │
                                │  TTL Cache (news/scores) │
                                │  Railway hosting         │
                                └──────────────────────────┘
```

---

## Backtesting

The `/backtest/{symbol}` endpoint correlates sentiment scores against next-day stock returns:

1. Pull 30 days of headlines via NewsAPI
2. Score each day's headlines, compute a daily aggregate sentiment score (range: -1 to +1)
3. Pull price data for the same period via yfinance
4. Compute next-day return for each trading day
5. Calculate Pearson correlation between same-day sentiment and next-day return
6. Identify the strongest sentiment-driven price move days

This transforms the project from a classification tool into **signal research** — the same methodology used by quantitative hedge funds analyzing alternative data.

---

## API Reference

The full interactive API is available at `/docs` (Swagger UI) once deployed.

```
POST /analyze                        Score a single headline
GET  /ticker/{symbol}                Live headlines + aggregate for a ticker
GET  /ticker/{symbol}/timeline       Daily sentiment breakdown (30 days)
GET  /backtest/{symbol}              Sentiment vs. price correlation
GET  /watchlist                      All watchlist tickers with sentiment
POST /watchlist                      Add ticker to watchlist
DELETE /watchlist/{symbol}           Remove ticker from watchlist
GET  /health                         Server status
```

---

## Local Setup

```bash
# Clone
git clone https://github.com/shravanA-git/nlp-finance-sentiment
cd nlp-finance-sentiment/backend

# Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your NewsAPI key from newsapi.org (free)

# Run
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
# Frontend: open frontend/index.html in browser
```

---

## Tech Stack

**Backend**: Python · FastAPI · PyTorch · HuggingFace Transformers · yfinance · NewsAPI · SQLite  
**Frontend**: HTML · Tailwind CSS · Chart.js  
**Deployment**: Railway (backend) · Netlify (frontend)

---

## Related Research

This project extends prior work in two earlier publications:

- [A Machine Learning Approach for Predicting Traffic Collision Severity](https://zenodo.org/...) — Zenodo, 2025
- [Rocket Flow: Understanding Drag Reduction through Geometry Analysis](https://zenodo.org/...) — Zenodo, 2025

---

## Future Work

- **Earnings call transcript analysis** — process 10-Q/10-K filings via SEC EDGAR API, score paragraph-by-paragraph
- **Portfolio-level sentiment** — aggregate sentiment across an entire holdings portfolio
- **Alert system** — notify when a watchlist ticker crosses a sentiment threshold
- **Fine-tuned model v2** — train on a larger, domain-specific corpus with more granular labels

---

## Author

**Shravan Anand** · Incoming Duke University Class of 2030 (CS + Economics)  
[GitHub](https://github.com/shravanA-git) · [Portfolio](https://bit.ly/SAnandportfolio)
