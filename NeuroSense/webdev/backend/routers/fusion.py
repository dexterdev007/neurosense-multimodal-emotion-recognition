from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.emotion_utils import SENTIMENT_ORDER, aggregate_probability_dict, map_emotion_to_sentiment
from utils.metadata_loader import load_modality_metadata
from utils.model_loader import get


router = APIRouter()

CORE_FUSION_MODALITIES = ("eeg", "speech", "face")
SUPPORTED_MODALITIES = CORE_FUSION_MODALITIES


class ModalityPayload(BaseModel):
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)


class FusionInput(BaseModel):
    eeg_probs: Optional[List[float]] = None
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


def _vector_from_payload(payload: ModalityPayload) -> np.ndarray:
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


def _available_modality_vectors(data: FusionInput) -> Dict[str, np.ndarray]:
    if data.modalities:
        vectors: Dict[str, np.ndarray] = {}
        for modality_name, payload in data.modalities.items():
            normalized_name = modality_name.strip().lower()
            if normalized_name not in SUPPORTED_MODALITIES:
                continue
            vectors[normalized_name] = _vector_from_payload(payload)

        if not vectors:
            raise HTTPException(status_code=400, detail="Fusion request did not include any supported modalities.")
        return vectors

    legacy_vectors = {
        "eeg": data.eeg_probs,
        "speech": data.speech_probs,
        "face": data.face_probs,
    }
    vectors = {
        modality_name: _vector_from_legacy_probs(values, f"{modality_name}_probs")
        for modality_name, values in legacy_vectors.items()
        if values is not None
    }
    if not vectors:
        raise HTTPException(status_code=400, detail="Fusion request did not include any modality probabilities.")
    return vectors


def _average_vectors(vectors: Dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.vstack(list(vectors.values()))
    averaged = stacked.mean(axis=0)
    total = float(averaged.sum())
    if total > 0:
        averaged = averaged / total
    return averaged


def _compute_average_baseline(vectors: Dict[str, np.ndarray]) -> dict:
    """Compute what simple averaging would predict — for transparency."""
    averaged = _average_vectors(vectors)
    baseline_label = SENTIMENT_ORDER[int(np.argmax(averaged))]
    baseline_probs = {
        sentiment: round(float(prob), 4)
        for sentiment, prob in zip(SENTIMENT_ORDER, averaged)
    }
    return {
        "prediction": baseline_label,
        "confidence": round(float(max(baseline_probs.values())), 4),
        "probabilities": baseline_probs,
    }


def _fusion_runtime_policy() -> dict:
    metadata = load_modality_metadata("fusion")
    adds_value = metadata.get("meta_model_adds_value")
    if adds_value is None:
        adds_value = True

    policy = {
        "meta_model_adds_value": bool(adds_value),
        "meta_model_disabled_reason": None,
    }

    if not policy["meta_model_adds_value"]:
        policy["meta_model_disabled_reason"] = (
            "The trained fusion meta-model did not outperform simple averaging on the saved validation metrics."
        )

    return policy


def _average_response(
    vectors: Dict[str, np.ndarray],
    active_modalities: tuple[str, ...],
    average_baseline: dict,
    policy: dict | None = None,
) -> dict:
    averaged_vector = _average_vectors(vectors)
    label = SENTIMENT_ORDER[int(np.argmax(averaged_vector))]
    probabilities = {
        sentiment: round(float(probability), 4)
        for sentiment, probability in zip(SENTIMENT_ORDER, averaged_vector)
    }

    response = {
        "modality": f"Fusion ({' + '.join(modality.upper() for modality in active_modalities)})",
        "prediction": label,
        "confidence": round(float(max(probabilities.values())), 4),
        "probabilities": probabilities,
        "model_used": f"Average sentiment aggregation ({len(active_modalities)} modalities)",
        "fusion_method": "average_aggregation",
        "feature_count": int(len(active_modalities) * len(SENTIMENT_ORDER)),
        "active_modalities": list(active_modalities),
        "baseline_average_prediction": average_baseline,
        "meta_agrees_with_baseline": True,
    }
    if policy and policy.get("meta_model_disabled_reason"):
        response["meta_model_disabled_reason"] = policy["meta_model_disabled_reason"]
    return response


@router.post("/predict")
def predict_fusion(data: FusionInput):
    try:
        vectors = _available_modality_vectors(data)
        active_modalities = tuple(vectors.keys())

        # Always compute the average baseline for comparison
        average_baseline = _compute_average_baseline(vectors)
        policy = _fusion_runtime_policy()

        # Use the trained late-fusion meta-model only when the exact trained
        # three-modality set is present. For partial subsets, aggregate the
        # available sentiment vectors directly.
        if set(active_modalities) == set(CORE_FUSION_MODALITIES) and policy["meta_model_adds_value"]:
            eeg_vector = vectors["eeg"]
            speech_vector = vectors["speech"]
            face_vector = vectors["face"]
            X_meta = np.concatenate([eeg_vector, speech_vector, face_vector]).reshape(1, -1)

            model = get("fusion", "model")
            encoder = get("fusion", "label_encoder")

            pred_idx = model.predict(X_meta)[0]
            label = encoder.inverse_transform([pred_idx])[0]

            probabilities = {}
            if hasattr(model, "predict_proba"):
                prob_arr = model.predict_proba(X_meta)[0]
                classes = encoder.inverse_transform(range(len(prob_arr)))
                probabilities = {cls: round(float(prob), 4) for cls, prob in zip(classes, prob_arr)}

            meta_agrees_with_avg = (label == average_baseline["prediction"])

            return {
                "modality": "Fusion (EEG + Speech + Face)",
                "prediction": label,
                "confidence": round(float(max(probabilities.values())), 4) if probabilities else None,
                "probabilities": probabilities,
                "model_used": "LogisticRegression (meta)",
                "fusion_method": "trained_meta_model",
                "feature_count": int(X_meta.shape[1]),
                "active_modalities": list(active_modalities),
                "baseline_average_prediction": average_baseline,
                "meta_agrees_with_baseline": meta_agrees_with_avg,
            }

        return _average_response(vectors, active_modalities, average_baseline, policy=policy)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fusion prediction failed: {exc}")
