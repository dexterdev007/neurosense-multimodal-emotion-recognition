from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routers import eeg, meg, mri, speech, face, fusion
from utils.model_loader import load_all_models

app = FastAPI(title="NeuroSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    try:
        load_all_models()
    except Exception as e:
        print(f"Warning: Models failed to load on startup: {e}. You may need to run notebooks first.")

app.include_router(eeg.router,    prefix="/api/eeg",    tags=["EEG"])
app.include_router(meg.router,    prefix="/api/meg",    tags=["MEG"])
app.include_router(mri.router,    prefix="/api/mri",    tags=["MRI"])
app.include_router(speech.router, prefix="/api/speech", tags=["Speech"])
app.include_router(face.router,   prefix="/api/face",   tags=["Face"])
app.include_router(fusion.router, prefix="/api/fusion", tags=["Fusion"])

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "NeuroSense API running"}

@app.get("/api/models/info")
def models_info():
    return {
        "eeg":    "SVM RBF — DEAP-derived EEG CSV (real dataset, cleanest module)",
        "meg":    "SVM RBF — Synthetic MEG features (NOT real MEG data — pipeline demo only)",
        "mri":    "SVM + PCA — Brain Tumor MRI, Kaggle (real dataset)",
        "speech": "SVM RBF — RAVDESS speech (24 actors), actor-holdout + GroupKFold evaluation",
        "face":   "SVM + PCA — FER2013 (known train-test gap, tuned C & PCA)",
        "fusion": "Late-fusion probability pooling (defaults to simple average when the saved meta-model does not beat averaging)"
    }

# Serve frontend as static files (so single Render URL works)
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Frontend directory not found at {frontend_path}")
