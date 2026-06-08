"""
model.py
--------
Loads FinBERT locally at startup. Runs on CPU — HuggingFace Spaces
provides enough RAM (16GB) to handle this comfortably.

Swap MODEL_NAME for your own model once uploaded to HuggingFace Hub:
    MODEL_NAME = "shravan-anand/financial-sentiment-bert"
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from schemas import SentimentScore

MODEL_NAME = "ProsusAI/finbert"
# ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


class SentimentModel:
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model ready on {self.device}.")

    def predict(self, text: str) -> SentimentScore:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=128, padding="max_length"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        label_idx = int(np.argmax(probs))
        scores = {LABEL_MAP[i]: float(probs[i]) for i in range(3)}
        return SentimentScore(
            label=LABEL_MAP[label_idx],
            confidence=float(probs[label_idx]),
            scores=scores
        )

    def predict_batch(self, texts: list[str]) -> list[SentimentScore]:
        return [self.predict(t) for t in texts]


_instance: SentimentModel | None = None

def get_model() -> SentimentModel:
    global _instance
    if _instance is None:
        _instance = SentimentModel()
    return _instance
