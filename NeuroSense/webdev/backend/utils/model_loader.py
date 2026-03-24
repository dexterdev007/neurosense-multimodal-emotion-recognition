import joblib, os

ARTIFACTS = {}

ARTIFACT_MAP = {
    "eeg":    ["model", "scaler", "label_encoder"],
    "meg":    ["model", "scaler", "label_encoder"],
    "mri":    ["model", "scaler", "pca", "label_encoder"],
    "speech": ["model", "scaler", "label_encoder"],
    "face":   ["model", "scaler", "pca", "label_encoder"],
    "fusion": ["model", "label_encoder"],
}

def load_all_models():
    # Try multiple base paths (handles different working directories)
    possible_bases = [
        os.path.join(os.path.dirname(__file__), "../../../artifacts"),
        os.path.join(os.path.dirname(__file__), "../../artifacts"),
        "artifacts",
        "../artifacts",
    ]
    
    base = None
    for candidate in possible_bases:
        if os.path.exists(candidate):
            base = os.path.abspath(candidate)
            break
    
    if not base:
        raise RuntimeError(
            "Cannot find /artifacts directory. "
            "Run the training notebooks first to generate .pkl files."
        )
    
    print(f"[NeuroSense] Loading artifacts from: {base}")
    
    for modality, keys in ARTIFACT_MAP.items():
        ARTIFACTS[modality] = {}
        for key in keys:
            filename = f"{modality}_{key}.pkl"
            filepath = os.path.join(base, modality, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Missing artifact: {filepath}\\n"
                    f"Run notebooks/0{list(ARTIFACT_MAP.keys()).index(modality)+1}_*.ipynb first."
                )
            ARTIFACTS[modality][key] = joblib.load(filepath)
            print(f"  Loaded: {modality}/{filename}")
    
    print("[NeuroSense] All artifacts loaded successfully.\\n")

def get(modality: str, key: str):
    if modality not in ARTIFACTS:
        raise KeyError(f"Modality '{modality}' not loaded. Check startup logs.")
    if key not in ARTIFACTS[modality]:
        raise KeyError(f"Key '{key}' not found for modality '{modality}'.")
    return ARTIFACTS[modality][key]
