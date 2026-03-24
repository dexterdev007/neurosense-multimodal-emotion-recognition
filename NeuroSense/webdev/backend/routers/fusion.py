from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.emotion_utils import SENTIMENT_ORDER, aggregate_probability_dict, map_emotion_to_sentiment
from utils.model_loader import get


router = APIRouter()


class ModalityPayload(BaseModel):
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)


class FusionInput(BaseModel):
    eeg_probs: Optional[List[float]] = None
    meg_probs: Optional[List[float]] = None
    speech_probs: Optional[List[float]] = None
    face_probs: Optional[List[float]] = None
    modalities: Dict[str, ModalityPayload] = Field(default_factory=dict)


def _prediction_fallback_vector(prediction: Optional[str], confidence: Optional[float]) -> np.ndarray:
    sentiment = map_emotion_to_sentiment(prediction or "NEUTRAL")
    clipped_confidence = float(confidence if confidence is not None else 1.0)
    clipped_confidence = min(max(clipped_confidence, 1.0 / len(SENTIMENT_ORDER)), 1.0)
    remainder = (1.0 - clipped_confidence) / (len(SENTIMENT_ORDER) - 1)

    vector = np.full(len(SENTIMENT_ORDER), remainder, dtype=np.float64)
    vector[SENTIMENT_ORDER.index(sentiment)] = clipped_confidence
    return vector


def _vector_from_modalities(data: FusionInput, modality_name: str) -> np.ndarray:
    payload = data.modalities.get(modality_name)
    if payload is None:
        raise HTTPException(status_code=400, detail=f"Missing modality '{modality_name}' in fusion request.")

    if payload.probabilities:
        return aggregate_probability_dict(payload.probabilities, order=SENTIMENT_ORDER)

    return _prediction_fallback_vector(payload.prediction, payload.confidence)


def _vector_from_legacy_probs(values: Optional[List[float]], field_name: str) -> np.ndarray:
    if values is None:
        raise HTTPException(status_code=400, detail=f"Missing field '{field_name}' in fusion request.")
    if len(values) != len(SENTIMENT_ORDER):
        raise HTTPException(
            status_code=400,
            detail=f"Field '{field_name}' must contain exactly {len(SENTIMENT_ORDER)} probabilities.",
        )

    vector = np.asarray(values, dtype=np.float64)
    total = float(vector.sum())
    if total > 0:
        vector = vector / total
    return vector


@router.post("/predict")
def predict_fusion(data: FusionInput):
    try:
        if data.modalities:
            eeg_vector = _vector_from_modalities(data, "eeg")
            meg_vector = _vector_from_modalities(data, "meg")
            speech_vector = _vector_from_modalities(data, "speech")
            face_vector = _vector_from_modalities(data, "face")
        else:
            eeg_vector = _vector_from_legacy_probs(data.eeg_probs, "eeg_probs")
            meg_vector = _vector_from_legacy_probs(data.meg_probs, "meg_probs")
            speech_vector = _vector_from_legacy_probs(data.speech_probs, "speech_probs")
            face_vector = _vector_from_legacy_probs(data.face_probs, "face_probs")

        X_meta = np.concatenate([eeg_vector, meg_vector, speech_vector, face_vector]).reshape(1, -1)

        model = get("fusion", "model")
        encoder = get("fusion", "label_encoder")

        pred_idx = model.predict(X_meta)[0]
        label = encoder.inverse_transform([pred_idx])[0]

        probabilities = {}
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(X_meta)[0]
            classes = encoder.inverse_transform(range(len(prob_arr)))
            probabilities = {cls: round(float(prob), 4) for cls, prob in zip(classes, prob_arr)}

        return {
            "modality": "Fusion (EEG + MEG + Speech + Face)",
            "prediction": label,
            "confidence": round(float(max(probabilities.values())), 4) if probabilities else None,
            "probabilities": probabilities,
            "model_used": "LogisticRegression (meta)",
            "feature_count": int(X_meta.shape[1]),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fusion prediction failed: {exc}")
