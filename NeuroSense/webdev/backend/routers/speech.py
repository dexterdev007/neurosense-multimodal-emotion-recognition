from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.model_loader import get
from utils.preprocessors import preprocess_speech

router = APIRouter()

@router.post("/predict")
async def predict_speech(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        model   = get("speech", "model")
        scaler  = get("speech", "scaler")
        encoder = get("speech", "label_encoder")
        
        X       = preprocess_speech(contents)
        X_sc    = scaler.transform(X)
        
        pred_idx = model.predict(X_sc)[0]
        label    = encoder.inverse_transform([pred_idx])[0]
        
        probs = {}
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(X_sc)[0]
            classes  = encoder.inverse_transform(range(len(prob_arr)))
            probs    = {c: round(float(p), 4) for c, p in zip(classes, prob_arr)}
        
        return {
            "modality":      "Speech",
            "prediction":    label,
            "confidence":    round(float(max(probs.values())), 4) if probs else None,
            "probabilities": probs,
            "model_used":    type(model).__name__
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech prediction failed: {str(e)}")
