# NLP Finance Sentiment App

## Project Goal
Full-stack web app that classifies financial headlines as Positive / Neutral / Negative using a fine-tuned BERT model. Built as a portfolio demo for Duke University CS + Economics recruiting.

## Owner
Shravan Anand — incoming Duke freshman, CS + Econ. Python/ML background. Published researcher (Zenodo). Comfort level: strong in Python/ML, learning web dev.

## Target Stack
- **Backend**: FastAPI (Python) — serves model inference via REST API
- **Frontend**: React + Tailwind — clean, modern UI
- **Model**: ProsusAI/finbert (HuggingFace) as default; swap for custom model later
- **Data**: yfinance for live news headlines
- **Deploy**: HuggingFace Spaces (Docker) or Railway

## Desired Features (priority order)
1. Single headline input → sentiment label + confidence bar
2. Live ticker mode → fetch recent headlines, score each, show aggregate
3. Sentiment timeline chart (last 30 days of headlines for a ticker)
4. Watchlist (track multiple tickers)

## Current State
- `app.py`: working Gradio demo already deployed on HuggingFace Spaces
- `save_model.py`: training script that saves model weights
- `requirements.txt`: dependencies (fixed for Python 3.13)
- `DEPLOY.md`: deployment guide

## Preferences
- Keep code clean and well-commented — this is a portfolio piece
- Prefer simplicity over cleverness
- Mobile-responsive UI
- No TypeScript for now — plain React is fine
