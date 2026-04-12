from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


ARTIFACTS: dict[str, dict[str, Any]] = {}

ARTIFACT_MAP = {
    "eeg": ["model", "scaler", "label_encoder"],
    "meg": ["model", "scaler", "label_encoder"],
    "mri": ["model", "scaler", "pca", "label_encoder"],
    "speech": ["model", "scaler", "label_encoder"],
    "face": ["model", "scaler", "pca", "label_encoder"],
    "fusion": ["model", "label_encoder"],
}


def _artifact_base_candidates() -> list[Path]:
    """Return the possible artifact folders for different working-directory layouts."""
    current_file = Path(__file__).resolve()
    return [
        current_file.parents[3] / "artifacts",
        current_file.parents[2] / "artifacts",
        Path.cwd() / "artifacts",
        Path.cwd().parent / "artifacts",
    ]


def _find_artifact_base() -> Path:
    """Pick the first artifact folder that actually exists on disk."""
    for candidate in _artifact_base_candidates():
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot find the artifacts directory. Run the training notebooks first to generate the .pkl files."
    )


def _artifact_path(base_dir: Path, modality: str, key: str) -> Path:
    return base_dir / modality / f"{modality}_{key}.pkl"


def load_all_models() -> None:
    """
    Load every saved model artifact into memory.

    The backend does this once during startup so later prediction requests can reuse
    the already-loaded objects.
    """
    base_dir = _find_artifact_base()
    print(f"[NeuroSense] Loading artifacts from: {base_dir}")

    for modality_index, (modality, keys) in enumerate(ARTIFACT_MAP.items(), start=1):
        ARTIFACTS[modality] = {}
        for key in keys:
            artifact_path = _artifact_path(base_dir, modality, key)
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Missing artifact: {artifact_path}\n"
                    f"Run notebooks/{modality_index:02d}_*.ipynb first."
                )

            ARTIFACTS[modality][key] = joblib.load(artifact_path)
            print(f"  Loaded: {modality}/{artifact_path.name}")

    print("[NeuroSense] All artifacts loaded successfully.\n")


def get(modality: str, key: str) -> Any:
    """Fetch one already-loaded artifact, such as `model` or `scaler`."""
    if modality not in ARTIFACTS:
        raise KeyError(f"Modality '{modality}' not loaded. Check startup logs.")
    if key not in ARTIFACTS[modality]:
        raise KeyError(f"Key '{key}' not found for modality '{modality}'.")
    return ARTIFACTS[modality][key]
