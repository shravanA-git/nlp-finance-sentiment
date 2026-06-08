"""
model.py
--------
Runs inference via the HuggingFace Inference API instead of loading weights
locally. This uses ~0 RAM on the server — the model runs on HuggingFace's
infrastructure and we call it over HTTP.

Requires a free HuggingFace token (HF_TOKEN env variable).
Get one at: huggingface.co/settings/tokens (read access is enough)

To swap in your own trained model, change HF_MODEL_ID to your repo name:
    HF_MODEL_ID = "shravan-anand/financial-sentiment-bert"
"""

import os
import requests
import time
from schemas import SentimentScore
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_MODEL_ID = "ProsusAI/finbert"
HF_API_URL  = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# FinBERT label order from HuggingFace API response
LABEL_NORM = {"positive": "positive", "negative": "negative", "neutral": "neutral",
              "POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral",
              "LABEL_0": "positive", "LABEL_1": "negative", "LABEL_2": "neutral"}


class SentimentModel:
    """
    Calls HuggingFace Inference API for predictions.
    No local model weights — works on any free hosting tier.
    """

    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        print(f"Model: {HF_MODEL_ID} via HuggingFace Inference API")
        # Warm up — first call loads the model on HF's side (~20s cold start)
        try:
            self._call_api("Market sentiment test.")
            print("HuggingFace Inference API ready.")
        except Exception as e:
            print(f"Warm-up note: {e} (will retry on first request)")

    def _call_api(self, text: str, retries: int = 3) -> list[dict]:
        """Call HF Inference API with retry on model loading (503)."""
        for attempt in range(retries):
            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 503:
                # Model is loading on HF side — wait and retry
                wait = int(response.headers.get("X-Wait-For-Model", "20"))
                print(f"Model loading on HF, waiting {wait}s...")
                time.sleep(min(wait, 20))
                continue
            response.raise_for_status()
        raise RuntimeError(f"HF API failed after {retries} attempts")

    def predict(self, text: str) -> SentimentScore:
        """Run inference on a single text string."""
        raw = self._call_api(text)

        # HF returns [[{label, score}, ...]] — unwrap
        if isinstance(raw, list) and isinstance(raw[0], list):
            raw = raw[0]

        scores_raw = {item["label"]: item["score"] for item in raw}

        # Normalize labels to lowercase consistent names
        scores = {}
        for raw_label, score in scores_raw.items():
            normalized = LABEL_NORM.get(raw_label, raw_label.lower())
            scores[normalized] = float(score)

        # Ensure all three keys exist
        for key in ("positive", "neutral", "negative"):
            scores.setdefault(key, 0.0)

        label      = max(scores, key=scores.get)
        confidence = scores[label]

        return SentimentScore(label=label, confidence=confidence, scores=scores)

    def predict_batch(self, texts: list[str]) -> list[SentimentScore]:
        return [self.predict(t) for t in texts]


_model_instance: SentimentModel | None = None


def get_model() -> SentimentModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentModel()
    return _model_instance
