"""
model.py
--------
Loads the FinBERT model once at startup and exposes a clean predict interface.

Model: ProsusAI/finbert — BERT fine-tuned on financial text, 3-class sentiment.
Swap MODEL_NAME for your own HuggingFace model once you upload it.

The SentimentModel class is instantiated once in main.py and injected as a
FastAPI dependency, so the 400MB model only loads once per server process.
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from schemas import SentimentScore

# Swap this once you upload your own trained model to HuggingFace Hub:
#   MODEL_NAME = "shravan-anand/financial-sentiment-bert"
MODEL_NAME = "ProsusAI/finbert"

# FinBERT uses a different label order than our original training code.
# ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
# We normalize to a consistent internal format.
FINBERT_LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


class SentimentModel:
    """
    Wraps FinBERT for single and batch inference.
    Load once, reuse across all requests.
    """

    def __init__(self):
        print(f"Loading model: {MODEL_NAME} ...")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model ready on {self.device}.")

    def predict(self, text: str) -> SentimentScore:
        """Run inference on a single text string."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        label_idx = int(np.argmax(probs))

        # Build a clean scores dict using our normalized label names
        scores = {FINBERT_LABEL_MAP[i]: float(probs[i]) for i in range(3)}
        label = FINBERT_LABEL_MAP[label_idx]
        confidence = float(probs[label_idx])

        return SentimentScore(label=label, confidence=confidence, scores=scores)

    def predict_batch(self, texts: list[str]) -> list[SentimentScore]:
        """
        Run inference on a list of texts.
        For CPU this is a simple loop; on GPU you'd batch them for speed.
        """
        return [self.predict(t) for t in texts]


# Singleton — imported and reused in main.py
_model_instance: SentimentModel | None = None


def get_model() -> SentimentModel:
    """FastAPI dependency: returns the cached model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentModel()
    return _model_instance
