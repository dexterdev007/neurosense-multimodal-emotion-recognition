from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _artifact_base_candidates() -> list[Path]:
    current_file = Path(__file__).resolve()
    return [
        current_file.parents[3] / "artifacts",
        Path.cwd() / "artifacts",
        Path.cwd().parent / "artifacts",
    ]


def load_modality_metadata(modality: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for artifacts_dir in _artifact_base_candidates():
        if not artifacts_dir.exists():
            continue

        root_metadata_path = artifacts_dir / "model_metadata.json"
        if root_metadata_path.exists():
            with root_metadata_path.open() as handle:
                root_metadata = json.load(handle)
            metadata.update(root_metadata.get("modalities", {}).get(modality, {}))

        modality_metadata_path = artifacts_dir / modality / f"{modality}_metadata.json"
        if modality_metadata_path.exists():
            with modality_metadata_path.open() as handle:
                metadata.update(json.load(handle))

        if metadata:
            break

    return metadata
