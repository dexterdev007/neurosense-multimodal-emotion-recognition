from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
from utils.model_loader import get
from utils.preprocessors import preprocess_mri_image

router = APIRouter()

@router.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        model   = get("mri", "model")
        scaler  = get("mri", "scaler")
        pca     = get("mri", "pca")
        encoder = get("mri", "label_encoder")
        
        X       = preprocess_mri_image(contents)
        X_sc    = scaler.transform(X)
        X_pca   = pca.transform(X_sc)
        
        pred_idx = model.predict(X_pca)[0]
        label    = encoder.inverse_transform([pred_idx])[0]
        
        probs = {}
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(X_pca)[0]
            classes  = encoder.inverse_transform(range(len(prob_arr)))
            probs    = {c: round(float(p), 4) for c, p in zip(classes, prob_arr)}
        
        return {
            "modality":      "MRI",
            "prediction":    label,
            "confidence":    round(float(max(probs.values())), 4) if probs else None,
            "probabilities": probs,
            "model_used":    type(model).__name__,
            "filename":      file.filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MRI prediction failed: {str(e)}")
