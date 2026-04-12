from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

CURRENT_FILE = Path(__file__).resolve()
NOTEBOOKS_DIR = CURRENT_FILE.parent
BACKEND_DIR = CURRENT_FILE.parents[1] / "webdev" / "backend"

for import_path in (NOTEBOOKS_DIR, BACKEND_DIR):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from notebook_support import bootstrap_notebook, extract_feature_dataset
from utils.emotion_utils import map_emotion_to_sentiment
from utils.preprocessors import preprocess_speech


RAVDESS_EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


@dataclass(frozen=True)
class Record:
    path: Path
    label: str
    actor_id: str


def _parse_ravdess_record(path: Path, label_mode: str) -> Record | None:
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None

    modality_code, channel_code, emotion_code, *_unused, actor_code = parts
    if modality_code != "03" or channel_code != "01":
        return None

    emotion_label = RAVDESS_EMOTION_CODES.get(emotion_code)
    if emotion_label is None:
        return None

    if label_mode == "sentiment":
        label = map_emotion_to_sentiment(emotion_label)
    else:
        label = emotion_label

    return Record(path=path, label=label, actor_id=f"Actor_{actor_code}")


def collect_ravdess_records(speech_root: Path, label_mode: str) -> list[Record]:
    records: list[Record] = []
    for audio_path in sorted(speech_root.rglob("*.wav")):
        record = _parse_ravdess_record(audio_path, label_mode=label_mode)
        if record is not None:
            records.append(record)

    if not records:
        raise FileNotFoundError(f"No RAVDESS speech files found under {speech_root}")
    return records


def candidate_models(random_state: int) -> dict[str, object]:
    return {
        "rbf_svm": SVC(
            kernel="rbf",
            C=8.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=900,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
    }


def make_pipeline_for_cv(model: object) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", clone(model)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NeuroSense speech model on RAVDESS.")
    parser.add_argument(
        "--label-mode",
        choices=("sentiment", "emotion"),
        default="sentiment",
        help="Train on 3 sentiment classes or the original 8 RAVDESS emotions.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of actors reserved for the final actor-holdout test split.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write the trained artifacts and metadata back into NeuroSense/artifacts/speech.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for cross-validation. Use 1 inside constrained environments.",
    )
    args = parser.parse_args()

    env = bootstrap_notebook()
    datasets_dir = Path(env["datasets_dir"])
    artifacts_dir = Path(env["artifacts_dir"])
    cache_dir = Path(env["cache_dir"])
    random_state = int(env["random_state"])

    speech_root = datasets_dir / "speech"
    cache_path = cache_dir / f"speech_ravdess_{args.label_mode}_features.npz"
    records = collect_ravdess_records(speech_root, label_mode=args.label_mode)

    print(f"Collected {len(records)} RAVDESS speech files from {speech_root}")
    print(f"Label mode: {args.label_mode}")

    X, y_raw = extract_feature_dataset(
        [(record.path, record.label) for record in records],
        preprocess_speech,
        cache_path=cache_path,
        progress_interval=100,
    )
    groups = np.asarray([record.actor_id for record in records])

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    unique_groups = np.unique(groups)
    print(f"Actors: {len(unique_groups)} -> {list(unique_groups)}")
    print(f"Classes: {list(encoder.classes_)}")
    print(f"Feature matrix shape: {X.shape}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(X, y, groups=groups))

    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    train_groups = groups[train_index]
    test_groups = groups[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_actor_list = sorted(set(train_groups))
    test_actor_list = sorted(set(test_groups))
    group_cv = GroupKFold(n_splits=min(6, len(train_actor_list)))

    best_name = ""
    best_cv = -1.0
    best_metrics: dict[str, object] = {}
    final_model = None

    print("\nModel comparison")
    print("-" * 72)
    for name, model in candidate_models(random_state=random_state).items():
        cv_scores = cross_val_score(
            make_pipeline_for_cv(model),
            X_train,
            y_train,
            groups=train_groups,
            cv=group_cv,
            scoring="accuracy",
            n_jobs=int(args.n_jobs),
        )

        fitted_model = clone(model)
        fitted_model.fit(X_train_scaled, y_train)
        y_pred = fitted_model.predict(X_test_scaled)
        holdout_accuracy = accuracy_score(y_test, y_pred)

        print(
            f"{name:20s} cv_mean={cv_scores.mean():.4f} "
            f"cv_std={cv_scores.std():.4f} holdout={holdout_accuracy:.4f}"
        )

        if cv_scores.mean() > best_cv:
            best_cv = float(cv_scores.mean())
            best_name = name
            best_metrics = {
                "cv_scores": cv_scores,
                "holdout_accuracy": holdout_accuracy,
                "classification_report": classification_report(
                    y_test,
                    y_pred,
                    target_names=list(encoder.classes_),
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "train_actors": train_actor_list,
                "test_actors": test_actor_list,
            }
            final_model = fitted_model

    if final_model is None:
        raise RuntimeError("No speech model was trained.")

    print("\nBest model:", best_name)
    print(f"Cross-validation mean accuracy: {best_cv:.4f}")
    print(f"Actor-holdout accuracy: {best_metrics['holdout_accuracy']:.4f}")
    print("\nClassification report")
    print(best_metrics["classification_report"])

    # Train one final deployment model on all available actors using the selected recipe.
    deployment_scaler = StandardScaler()
    X_all_scaled = deployment_scaler.fit_transform(X)
    deployment_model = clone(candidate_models(random_state=random_state)[best_name])
    deployment_model.fit(X_all_scaled, y)

    logo_cv = GroupKFold(n_splits=min(8, len(unique_groups)))
    logo_scores = cross_val_score(
        make_pipeline_for_cv(candidate_models(random_state=random_state)[best_name]),
        X,
        y,
        groups=groups,
        cv=logo_cv,
        scoring="accuracy",
        n_jobs=int(args.n_jobs),
    )
    print(
        f"Deployment-estimate GroupKFold mean accuracy: "
        f"{logo_scores.mean():.4f} +/- {logo_scores.std():.4f}"
    )

    metadata = {
        "data_source": "Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)",
        "data_source_note": (
            "Audio-only speech clips from 24 actors. Labels are derived from the official RAVDESS "
            f"emotion codes and mapped to {args.label_mode} classes for this project."
        ),
        "dataset_name": "RAVDESS Audio Speech Actors 01-24",
        "evaluation_method": "Actor Holdout + GroupKFold cross-validation (speaker-aware)",
        "evaluation_note": (
            "Model selection uses speaker-aware GroupKFold on train actors, then reports a final "
            "actor-holdout score on unseen actors."
        ),
        "selected_model": best_name,
        "label_mode": args.label_mode,
        "output_classes": list(encoder.classes_),
        "actor_holdout_accuracy": round(float(best_metrics["holdout_accuracy"]), 4),
        "groupkfold_mean_accuracy": round(float(best_cv), 4),
        "groupkfold_std": round(float(np.std(best_metrics["cv_scores"])), 4),
        "all_actor_groupkfold_mean_accuracy": round(float(logo_scores.mean()), 4),
        "all_actor_groupkfold_std": round(float(logo_scores.std()), 4),
        "n_samples": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_actors": train_actor_list,
        "test_actors": test_actor_list,
        "cache_file": cache_path.name,
        "confusion_matrix": best_metrics["confusion_matrix"],
    }

    if args.save:
        speech_artifact_dir = artifacts_dir / "speech"
        speech_artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(deployment_model, speech_artifact_dir / "speech_model.pkl")
        joblib.dump(deployment_scaler, speech_artifact_dir / "speech_scaler.pkl")
        joblib.dump(encoder, speech_artifact_dir / "speech_label_encoder.pkl")

        with (speech_artifact_dir / "speech_metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)

        # Compatibility: older notebooks (including fusion) expect this cache filename.
        legacy_cache_path = cache_dir / "speech_features_deduped.npz"
        if cache_path.exists() and cache_path != legacy_cache_path:
            shutil.copyfile(cache_path, legacy_cache_path)

        print(f"\nSaved updated speech artifacts to {speech_artifact_dir}")


if __name__ == "__main__":
    main()
