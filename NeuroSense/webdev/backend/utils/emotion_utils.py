from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


SENTIMENT_ORDER = ("NEGATIVE", "NEUTRAL", "POSITIVE")


def _normalize_label(label: object) -> str:
    text = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def map_emotion_to_sentiment(label: object) -> str:
    normalized = _normalize_label(label)

    if normalized in {"negative", "neg"}:
        return "NEGATIVE"
    if normalized in {"neutral", "neu"}:
        return "NEUTRAL"
    if normalized in {"positive", "pos"}:
        return "POSITIVE"

    if any(token in normalized for token in ("happy", "surprise", "pleasant")) or normalized == "ps":
        return "POSITIVE"
    if "neutral" in normalized or normalized == "no_tumor":
        return "NEUTRAL"
    if any(
        token in normalized
        for token in ("angry", "disgust", "fear", "sad", "tumor", "glioma", "meningioma", "pituitary")
    ):
        return "NEGATIVE"

    return "NEUTRAL"


def aggregate_probabilities(
    labels: Sequence[object],
    probabilities: Sequence[float],
    order: Sequence[str] = SENTIMENT_ORDER,
) -> np.ndarray:
    totals = {sentiment: 0.0 for sentiment in order}
    for label, probability in zip(labels, probabilities):
        totals[map_emotion_to_sentiment(label)] += float(probability)

    vector = np.array([totals[sentiment] for sentiment in order], dtype=np.float64)
    total = float(vector.sum())
    if total > 0:
        vector /= total
    return vector


def aggregate_probability_dict(
    probability_map: Mapping[object, float],
    order: Sequence[str] = SENTIMENT_ORDER,
) -> np.ndarray:
    return aggregate_probabilities(list(probability_map.keys()), list(probability_map.values()), order=order)
