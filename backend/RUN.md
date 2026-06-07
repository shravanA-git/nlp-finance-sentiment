# Running the Backend

## First-time setup (do once)

```bash
cd research/nlp-finance-demo/backend

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Add your NewsAPI key
cp .env.example .env
# Open .env and paste your key from newsapi.org
```

## Run the server

```bash
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

First startup takes ~30 seconds while it downloads the FinBERT model (~440MB).
Subsequent starts are instant (model is cached in ~/.cache/huggingface).

## Explore the API

Once running, open your browser to:
  http://localhost:8000/docs        ← interactive API explorer (try every endpoint here)
  http://localhost:8000/redoc       ← clean reference docs

## Test key endpoints manually

```bash
# Single headline
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"headline": "Apple reports record quarterly earnings"}'

# Ticker sentiment
curl http://localhost:8000/ticker/AAPL

# 30-day timeline (requires NewsAPI key)
curl http://localhost:8000/ticker/AAPL/timeline

# Watchlist
curl http://localhost:8000/watchlist
```
