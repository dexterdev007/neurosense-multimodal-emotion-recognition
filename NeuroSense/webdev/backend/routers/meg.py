import warnings
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.metadata_loader import load_modality_metadata
from utils.model_loader import get
from utils.preprocessors import preprocess_meg

router = APIRouter()


class MEGInput(BaseModel):
    features: List[float]

@router.post("/predict")
def predict_meg(data: MEGInput):
    try:
        model   = get("meg", "model")
        scaler  = get("meg", "scaler")
        encoder = get("meg", "label_encoder")
        
        X = preprocess_meg(data.features)
        
        expected = scaler.n_features_in_
        if X.shape[1] != expected:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected} features, got {X.shape[1]}."
            )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X_scaled = scaler.transform(X)
        pred_idx = model.predict(X_scaled)[0]
        label    = encoder.inverse_transform([pred_idx])[0]
        
        probs = {}
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(X_scaled)[0]
            classes  = encoder.inverse_transform(range(len(prob_arr)))
            probs    = {c: round(float(p), 4) for c, p in zip(classes, prob_arr)}

        # Load metadata for data_source transparency
        metadata = load_modality_metadata("meg")
        data_source = metadata.get("data_source", "unknown")
        data_source_note = metadata.get(
            "data_source_note",
            (
                "Trained on synthetic MEG-like data — accuracy does not reflect real-world performance"
                if data_source == "synthetic"
                else ("Trained on real MEG data" if data_source == "real" else "MEG data source is not specified")
            ),
        )
            
        return {
            "modality":      "MEG",
            "prediction":    label,
            "confidence":    round(float(max(probs.values())), 4) if probs else None,
            "probabilities": probs,
            "model_used":    type(model).__name__,
            "feature_count": int(X.shape[1]),
            "data_source":   data_source,
            "data_source_note": data_source_note,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MEG prediction failed: {str(e)}")
