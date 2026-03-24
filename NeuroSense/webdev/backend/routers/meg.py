from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
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
        
        X_scaled = scaler.transform(X)
        pred_idx = model.predict(X_scaled)[0]
        label    = encoder.inverse_transform([pred_idx])[0]
        
        probs = {}
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(X_scaled)[0]
            classes  = encoder.inverse_transform(range(len(prob_arr)))
            probs    = {c: round(float(p), 4) for c, p in zip(classes, prob_arr)}
            
        return {
            "modality":      "MEG",
            "prediction":    label,
            "confidence":    round(float(max(probs.values())), 4) if probs else None,
            "probabilities": probs,
            "model_used":    type(model).__name__,
            "feature_count": int(X.shape[1])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MEG prediction failed: {str(e)}")
