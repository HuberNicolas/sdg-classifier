"""Predict the SDGs of texts with a trained model."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SDGClassifier:
    def __init__(self, model_dir: Path | str, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.max_length = max_length

    @torch.no_grad()
    def probabilities(self, texts: list[str]) -> list[dict[str, float]]:
        batch = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        probs = torch.sigmoid(self.model(**batch).logits)
        labels = self.model.config.id2label
        return [{labels[i]: round(p, 4) for i, p in enumerate(row.tolist())} for row in probs]

    def predict(self, texts: list[str], threshold: float = 0.5) -> list[list[str]]:
        return [[sdg for sdg, p in probs.items() if p >= threshold] for probs in self.probabilities(texts)]
