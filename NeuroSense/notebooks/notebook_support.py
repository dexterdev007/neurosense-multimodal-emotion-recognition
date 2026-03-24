from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "NeuroSense" / "webdev" / "backend").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current notebook session.")


def bootstrap_notebook(project_root: Path | None = None, random_state: int = 42) -> dict[str, Path | int]:
    root = find_project_root(project_root)
    notebooks_dir = root / "NeuroSense" / "notebooks"
    backend_dir = root / "NeuroSense" / "webdev" / "backend"
    datasets_dir = root / "NeuroSense" / "datasets"
    artifacts_dir = root / "NeuroSense" / "artifacts"
    cache_dir = artifacts_dir / "cache"
    tmp_dir = root / "tmp"
    mpl_dir = tmp_dir / "mplconfig"
    xdg_dir = tmp_dir / "xdg-cache"

    for directory in (cache_dir, mpl_dir, xdg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))

    for path in (notebooks_dir, backend_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return {
        "project_root": root,
        "notebooks_dir": notebooks_dir,
        "backend_dir": backend_dir,
        "datasets_dir": datasets_dir,
        "artifacts_dir": artifacts_dir,
        "cache_dir": cache_dir,
        "random_state": random_state,
    }


def resolve_mri_directories(datasets_dir: Path) -> tuple[Path, Path]:
    candidates = [
        datasets_dir / "mri" / "archive",
        datasets_dir / "mri" / "mri",
    ]

    for base_dir in candidates:
        train_dir = base_dir / "Training"
        test_dir = base_dir / "Testing"
        if train_dir.exists() and test_dir.exists():
            return train_dir, test_dir

    raise FileNotFoundError("Could not find MRI Training/Testing directories in NeuroSense/datasets/mri.")


def sample_records_by_label(
    records: Iterable[tuple[Path, str]],
    per_label: int | None = None,
    seed: int = 42,
) -> list[tuple[Path, str]]:
    records = list(records)
    if per_label is None:
        return sorted(records, key=lambda item: (item[1], str(item[0])))

    rng = np.random.default_rng(seed)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path, label in records:
        grouped[label].append(path)

    sampled: list[tuple[Path, str]] = []
    for label, paths in grouped.items():
        if len(paths) <= per_label:
            chosen = paths
        else:
            indices = rng.choice(len(paths), size=per_label, replace=False)
            chosen = [paths[index] for index in np.sort(indices)]
        sampled.extend((path, label) for path in chosen)

    return sorted(sampled, key=lambda item: (item[1], str(item[0])))


def collect_labeled_image_paths(
    split_dir: Path,
    label_fn: Callable[[str], str | None],
) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = label_fn(class_dir.name)
        if label is None:
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append((image_path, label))
    return records


def collect_audio_paths(
    audio_root: Path,
    label_fn: Callable[[str], str | None],
) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for audio_path in sorted(audio_root.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        label = label_fn(audio_path.parent.name)
        if label is not None:
            records.append((audio_path, label))
    return records


def extract_feature_dataset(
    records: Iterable[tuple[Path, str]],
    feature_fn: Callable[[bytes], np.ndarray],
    cache_path: Path | None = None,
    dtype: np.dtype = np.float32,
    progress_interval: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"]

    records = list(records)
    features: list[np.ndarray] = []
    labels: list[str] = []

    for index, (path, label) in enumerate(records, start=1):
        with path.open("rb") as handle:
            feature_vector = feature_fn(handle.read()).reshape(-1).astype(dtype, copy=False)
        features.append(feature_vector)
        labels.append(label)

        if progress_interval and (index == 1 or index % progress_interval == 0 or index == len(records)):
            print(f"Processed {index}/{len(records)} files")

    if not features:
        raise ValueError("No training samples were collected from the dataset.")

    X = np.vstack(features)
    y = np.asarray(labels)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y


def label_counts(labels: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items()))
