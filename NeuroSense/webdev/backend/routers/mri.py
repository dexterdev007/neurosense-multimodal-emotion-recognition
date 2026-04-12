import warnings

from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.metadata_loader import load_modality_metadata
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X_pca = pca.transform(X_sc)
        
        pred_idx = model.predict(X_pca)[0]
        label    = encoder.inverse_transform([pred_idx])[0]
        
        probs = {}
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(X_pca)[0]
            classes  = encoder.inverse_transform(range(len(prob_arr)))
            probs    = {c: round(float(p), 4) for c, p in zip(classes, prob_arr)}
        
        metadata = load_modality_metadata("mri")
        
        return {
            "modality":      "MRI",
            "prediction":    label,
            "confidence":    round(float(max(probs.values())), 4) if probs else None,
            "probabilities": probs,
            "model_used":    type(model).__name__,
            "filename":      file.filename,
            "data_source":   metadata.get("data_source"),
            "data_source_note": metadata.get("data_source_note"),
            "evaluation_method": metadata.get("evaluation_method")
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MRI prediction failed: {str(e)}")
