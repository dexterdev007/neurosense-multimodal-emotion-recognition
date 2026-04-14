from __future__ import annotations

"""
Small audit script for checking whether saved ML artifacts include transparency metadata.

This is useful for project demonstrations because it verifies that each modality keeps
basic notes about where the data came from and how the model was evaluated.
"""

import json
import sys
from pathlib import Path
from typing import Any


ARTIFACTS_DIR = Path(__file__).parent / "NeuroSense" / "artifacts"
MODALITIES = ["eeg", "speech", "face", "fusion"]
INTEGRITY_MARKERS = ["data_source", "evaluation_method", "data_source_note"]


def _load_metadata(name: str) -> tuple[dict[str, Any] | None, str]:
    """Load per-modality metadata, falling back to the root metadata file when needed."""
    modality_dir = ARTIFACTS_DIR / name
    modality_metadata_path = modality_dir / f"{name}_metadata.json"

    if modality_metadata_path.exists():
        with modality_metadata_path.open() as handle:
            return json.load(handle), f"PASS (Local {name}_metadata.json found)"

    root_metadata_path = ARTIFACTS_DIR / "model_metadata.json"
    if not root_metadata_path.exists():
        return None, f"FAIL (Missing {name}_metadata.json)"

    with root_metadata_path.open() as handle:
        root_metadata = json.load(handle)

    fallback_metadata = root_metadata.get("modalities", {}).get(name, {})
    if fallback_metadata:
        return fallback_metadata, "WARNING (Using generic model_metadata.json, local version preferred)"

    return None, "FAIL (No local or root metadata found)"


def audit_modality(name: str) -> bool:
    """Print an integrity summary for one modality and return whether it passes."""
    print(f"\n--- Auditing {name.upper()} ---")
    modality_dir = ARTIFACTS_DIR / name

    artifact_files = list(modality_dir.glob("*.pkl"))
    artifact_status = "PASS" if artifact_files else "FAIL (MISSING .PKL)"
    print(f"  [Artifacts]: {artifact_status} ({len(artifact_files)} files found)")

    metadata, metadata_status = _load_metadata(name)
    print(f"  [Metadata]:  {metadata_status}")
    if metadata is None:
        return False

    missing_markers = [marker for marker in INTEGRITY_MARKERS if marker not in metadata and f"{name}_{marker}" not in metadata]
    if missing_markers:
        print(f"  [Integrity]: FAIL (Missing markers: {', '.join(missing_markers)})")
        return False

    print("  [Integrity]: PASS (Honesty markers present)")
    return True


def main() -> None:
    if not ARTIFACTS_DIR.exists():
        print(f"Error: Artifacts directory not found at {ARTIFACTS_DIR}")
        sys.exit(1)

    all_passed = True
    for modality in MODALITIES:
        if not audit_modality(modality):
            all_passed = False

    print("\n" + "=" * 30)
    if all_passed:
        print("RESULT: ALL MODALITIES COMPLIANT ✅")
    else:
        print("RESULT: MISSING INTEGRITY DATA ❌")
        print("Action Required: Update notebooks and regenerate artifacts.")


if __name__ == "__main__":
    main()
