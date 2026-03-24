from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "NeuroSense" / "notebooks"


def notebook_cell(cell_type: str, source: str, cell_id: str) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    cell = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def write_notebook(filename: str, cells: list[tuple[str, str]]) -> None:
    slug = Path(filename).stem.lower().replace("_", "-")
    notebook = {
        "cells": [
            notebook_cell(cell_type, source, cell_id=f"{slug}-{index}")
            for index, (cell_type, source) in enumerate(cells, start=1)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Human Emotion (venv)",
                "language": "python",
                "name": "human-emotion-venv",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path = NOTEBOOK_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {path}")


BOOTSTRAP_CELL = dedent(
    """
    from pathlib import Path
    import sys

    def _find_project_root():
        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            if (candidate / "NeuroSense" / "webdev" / "backend").exists():
                return candidate
        raise FileNotFoundError("Could not locate the project root for this notebook.")

    PROJECT_ROOT = _find_project_root()
    NOTEBOOKS_DIR = PROJECT_ROOT / "NeuroSense" / "notebooks"
    BACKEND_DIR = PROJECT_ROOT / "NeuroSense" / "webdev" / "backend"

    for path in (NOTEBOOKS_DIR, BACKEND_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from notebook_support import bootstrap_notebook

    ctx = bootstrap_notebook(PROJECT_ROOT)
    DATASETS_DIR = ctx["datasets_dir"]
    ARTIFACTS_DIR = ctx["artifacts_dir"]
    CACHE_DIR = ctx["cache_dir"]
    RANDOM_STATE = ctx["random_state"]

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Datasets directory: {DATASETS_DIR}")
    print(f"Artifacts directory: {ARTIFACTS_DIR}")
    """
).strip()


EEG_NOTEBOOK = [
    (
        "markdown",
        dedent(
            """
            # 01. EEG Emotion Recognition

            Train a classical machine-learning model on the EEG emotion dataset and save the scaler, label encoder, and classifier artifacts used by the backend.
            """
        ).strip(),
    ),
    ("code", BOOTSTRAP_CELL),
    (
        "code",
        dedent(
            """
            import joblib
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
            from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
            from sklearn.svm import SVC

            eeg_path = DATASETS_DIR / "eeg" / "eeg" / "emotions.csv"
            if not eeg_path.exists():
                raise FileNotFoundError(f"Missing EEG dataset: {eeg_path}")

            df = pd.read_csv(eeg_path)
            if "label" not in df.columns:
                raise ValueError("Expected a 'label' column in the EEG dataset.")

            feature_columns = [column for column in df.columns if column != "label"]
            X = df[feature_columns].select_dtypes(include=[np.number]).fillna(0.0)
            y = df["label"].astype(str)

            print("Dataset shape:", df.shape)
            print("Label counts:", y.value_counts().to_dict())
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.countplot(x=y, ax=axes[0], order=sorted(y.unique()), palette="viridis")
            axes[0].set_title("EEG label distribution")
            axes[0].tick_params(axis="x", rotation=20)

            axes[1].hist(X.iloc[:, 0], bins=20, color="#2d8fdd")
            axes[1].set_title(f"Distribution of {feature_columns[0]}")
            plt.tight_layout()
            plt.show()

            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X.values,
                y_enc,
                test_size=0.2,
                stratify=y_enc,
                random_state=RANDOM_STATE,
            )

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            candidate_models = {
                "svm_rbf": SVC(kernel="rbf", C=10.0, probability=True, random_state=RANDOM_STATE),
                "random_forest": RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1),
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            model_rows = []
            fitted_models = {}

            for name, estimator in candidate_models.items():
                cv_scores = cross_val_score(
                    make_pipeline(StandardScaler(), estimator),
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="accuracy",
                    n_jobs=-1,
                )
                fitted_estimator = estimator.fit(X_train_sc, y_train)
                fitted_models[name] = fitted_estimator
                test_accuracy = accuracy_score(y_test, fitted_estimator.predict(X_test_sc))
                model_rows.append(
                    {
                        "model": name,
                        "cv_mean": round(float(cv_scores.mean()), 4),
                        "cv_std": round(float(cv_scores.std()), 4),
                        "test_accuracy": round(float(test_accuracy), 4),
                    }
                )

            results_df = pd.DataFrame(model_rows).sort_values(["test_accuracy", "cv_mean"], ascending=False)
            best_name = results_df.iloc[0]["model"]
            best_model = fitted_models[best_name]
            results_df
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            y_pred = best_model.predict(X_test_sc)
            print(f"Selected model: {best_name}")
            print(classification_report(y_test, y_pred, target_names=le.classes_))

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title("EEG confusion matrix")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.show()

            y_test_bin = label_binarize(y_test, classes=list(range(len(le.classes_))))
            y_score = best_model.predict_proba(X_test_sc)

            plt.figure(figsize=(7, 5))
            for index, class_name in enumerate(le.classes_):
                fpr, tpr, _ = roc_curve(y_test_bin[:, index], y_score[:, index])
                plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc(fpr, tpr):.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.title("EEG one-vs-rest ROC curves")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            artifact_dir = ARTIFACTS_DIR / "eeg"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(best_model, artifact_dir / "eeg_model.pkl")
            joblib.dump(scaler, artifact_dir / "eeg_scaler.pkl")
            joblib.dump(le, artifact_dir / "eeg_label_encoder.pkl")

            sample_probs = best_model.predict_proba(X_test_sc[:3])
            print("Saved EEG artifacts to:", artifact_dir)
            print("Sample probabilities:")
            print(pd.DataFrame(sample_probs, columns=le.classes_).round(4))
            """
        ).strip(),
    ),
]


MEG_NOTEBOOK = [
    (
        "markdown",
        dedent(
            """
            # 02. MEG Emotion Recognition

            Generate a synthetic MEG feature dataset when it is missing, train an SVM classifier, and save backend-compatible artifacts.
            """
        ).strip(),
    ),
    ("code", BOOTSTRAP_CELL),
    (
        "code",
        dedent(
            """
            import joblib
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
            from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
            from sklearn.svm import SVC

            meg_path = DATASETS_DIR / "meg" / "meg_features.csv"
            meg_path.parent.mkdir(parents=True, exist_ok=True)

            if not meg_path.exists():
                rng = np.random.default_rng(RANDOM_STATE)
                n_samples = 1000
                features = rng.normal(size=(n_samples, 50))
                labels = rng.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"], size=n_samples, p=[0.34, 0.33, 0.33])

                for index, label in enumerate(labels):
                    if label == "POSITIVE":
                        features[index, :10] += 1.5
                    elif label == "NEGATIVE":
                        features[index, 10:20] += 1.5
                    else:
                        features[index, 20:30] += 0.5

                meg_df = pd.DataFrame(features, columns=[f"meg_f{idx}" for idx in range(features.shape[1])])
                meg_df["label"] = labels
                meg_df.to_csv(meg_path, index=False)
                print(f"Generated synthetic MEG dataset at {meg_path}")

            df = pd.read_csv(meg_path)
            X = df.drop(columns=["label"]).astype(np.float32)
            y = df["label"].astype(str)

            print("Dataset shape:", df.shape)
            print("Label counts:", y.value_counts().to_dict())
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X.values,
                y_enc,
                test_size=0.2,
                stratify=y_enc,
                random_state=RANDOM_STATE,
            )

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            model = SVC(kernel="rbf", C=10.0, probability=True, random_state=RANDOM_STATE)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(
                make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, probability=True, random_state=RANDOM_STATE)),
                X_train,
                y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )

            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            print(f"MEG test accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"MEG CV mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title("MEG confusion matrix")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.show()

            y_test_bin = label_binarize(y_test, classes=list(range(len(le.classes_))))
            y_score = model.predict_proba(X_test_sc)

            plt.figure(figsize=(7, 5))
            for index, class_name in enumerate(le.classes_):
                fpr, tpr, _ = roc_curve(y_test_bin[:, index], y_score[:, index])
                plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc(fpr, tpr):.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.title("MEG one-vs-rest ROC curves")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            artifact_dir = ARTIFACTS_DIR / "meg"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, artifact_dir / "meg_model.pkl")
            joblib.dump(scaler, artifact_dir / "meg_scaler.pkl")
            joblib.dump(le, artifact_dir / "meg_label_encoder.pkl")

            print("Saved MEG artifacts to:", artifact_dir)
            """
        ).strip(),
    ),
]


MRI_NOTEBOOK = [
    (
        "markdown",
        dedent(
            """
            # 03. MRI Brain Tumor Classification

            Extract grayscale 64x64 image features that match the backend MRI preprocessor, then train a PCA + SVM pipeline and save the artifacts.
            """
        ).strip(),
    ),
    ("code", BOOTSTRAP_CELL),
    (
        "code",
        dedent(
            """
            import joblib
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            from sklearn.decomposition import PCA
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.svm import SVC

            from notebook_support import collect_labeled_image_paths, extract_feature_dataset, label_counts, resolve_mri_directories
            from utils.preprocessors import preprocess_mri_image

            def normalize_mri_label(folder_name):
                return "no_tumor" if folder_name.lower() == "notumor" else folder_name.lower()

            train_dir, test_dir = resolve_mri_directories(DATASETS_DIR)
            train_records = collect_labeled_image_paths(train_dir, normalize_mri_label)
            test_records = collect_labeled_image_paths(test_dir, normalize_mri_label)

            print("MRI training labels:", label_counts(label for _, label in train_records))
            print("MRI testing labels:", label_counts(label for _, label in test_records))

            X_train_raw, y_train_raw = extract_feature_dataset(
                train_records,
                preprocess_mri_image,
                cache_path=CACHE_DIR / "mri_train_features.npz",
                progress_interval=250,
            )
            X_test_raw, y_test_raw = extract_feature_dataset(
                test_records,
                preprocess_mri_image,
                cache_path=CACHE_DIR / "mri_test_features.npz",
                progress_interval=250,
            )

            print("MRI feature matrix:", X_train_raw.shape, X_test_raw.shape)
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            y_test = le.transform(y_test_raw)

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_raw)
            X_test_sc = scaler.transform(X_test_raw)

            pca = PCA(n_components=100, svd_solver="randomized", random_state=RANDOM_STATE)
            X_train_pca = pca.fit_transform(X_train_sc)
            X_test_pca = pca.transform(X_test_sc)

            model = SVC(kernel="rbf", C=5.0, probability=True, random_state=RANDOM_STATE)
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)

            print(f"MRI test accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            print("PCA explained variance:", round(float(np.sum(pca.explained_variance_ratio_)), 4))
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title("MRI confusion matrix")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.show()

            figure, axes = plt.subplots(1, 3, figsize=(12, 4))
            for axis, (sample, label) in zip(axes, zip(X_test_raw[:3], y_test_raw[:3])):
                axis.imshow(sample.reshape(64, 64), cmap="gray")
                axis.set_title(label)
                axis.axis("off")
            plt.tight_layout()
            plt.show()
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            try:
                import torch
                import torch.nn as nn
                from torchvision import models

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cnn_model = models.resnet18(weights=None)
                cnn_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                cnn_model.fc = nn.Linear(cnn_model.fc.in_features, len(le.classes_))
                cnn_model = cnn_model.to(device)
                print("Optional CNN baseline instantiated on:", device)
            except Exception as exc:
                print(f"Skipping optional CNN baseline: {exc}")
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            artifact_dir = ARTIFACTS_DIR / "mri"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, artifact_dir / "mri_model.pkl")
            joblib.dump(scaler, artifact_dir / "mri_scaler.pkl")
            joblib.dump(pca, artifact_dir / "mri_pca.pkl")
            joblib.dump(le, artifact_dir / "mri_label_encoder.pkl")

            print("Saved MRI artifacts to:", artifact_dir)
            """
        ).strip(),
    ),
]


SPEECH_NOTEBOOK = [
    (
        "markdown",
        dedent(
            """
            # 04. Speech Emotion Recognition

            Extract 162-dimensional speech features using the exact backend preprocessing logic, train an SVM, and save the artifacts used by the API.
            """
        ).strip(),
    ),
    ("code", BOOTSTRAP_CELL),
    (
        "code",
        dedent(
            """
            import joblib
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.svm import SVC

            from notebook_support import collect_audio_paths, extract_feature_dataset, label_counts, sample_records_by_label
            from utils.emotion_utils import map_emotion_to_sentiment
            from utils.preprocessors import preprocess_speech

            def normalize_speech_label(folder_name):
                normalized = folder_name.lower().replace("-", "_").replace(" ", "_")
                if "pleasant" in normalized or normalized.endswith("_ps") or normalized == "ps":
                    return "ps"
                for emotion in ("angry", "disgust", "fear", "happy", "neutral", "sad"):
                    if emotion in normalized:
                        return emotion
                return None

            speech_root = DATASETS_DIR / "speech"
            if not speech_root.exists():
                raise FileNotFoundError(f"Missing speech dataset directory: {speech_root}")

            MAX_FILES_PER_EMOTION = 150

            records = collect_audio_paths(speech_root, normalize_speech_label)
            records = sample_records_by_label(records, per_label=MAX_FILES_PER_EMOTION, seed=RANDOM_STATE)
            print("Speech label counts:", label_counts(label for _, label in records))
            print("Speech sentiment counts:", label_counts(map_emotion_to_sentiment(label) for _, label in records))

            X, y_raw = extract_feature_dataset(
                records,
                preprocess_speech,
                cache_path=CACHE_DIR / "speech_features.npz",
                progress_interval=100,
            )

            print("Speech feature matrix:", X.shape)
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            le = LabelEncoder()
            y = le.fit_transform(y_raw)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=RANDOM_STATE,
            )

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            model = SVC(kernel="rbf", C=5.0, probability=True, random_state=RANDOM_STATE)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(
                make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5.0, probability=True, random_state=RANDOM_STATE)),
                X_train,
                y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )

            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            print(f"Speech test accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"Speech CV mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="magma", xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title("Speech confusion matrix")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.show()

            feature_summary = pd.DataFrame(X[:5]).T.describe().T[["mean", "std"]].head(10)
            feature_summary
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            artifact_dir = ARTIFACTS_DIR / "speech"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, artifact_dir / "speech_model.pkl")
            joblib.dump(scaler, artifact_dir / "speech_scaler.pkl")
            joblib.dump(le, artifact_dir / "speech_label_encoder.pkl")

            print("Saved speech artifacts to:", artifact_dir)
            """
        ).strip(),
    ),
]


FACE_NOTEBOOK = [
    (
        "markdown",
        dedent(
            """
            # 05. Facial Expression Recognition

            Extract 2308-dimensional face features that match the backend image preprocessor, train a PCA + SVM sentiment model, and save the artifacts.
            """
        ).strip(),
    ),
    ("code", BOOTSTRAP_CELL),
    (
        "code",
        dedent(
            """
            import joblib
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            from sklearn.decomposition import PCA
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.svm import SVC

            from notebook_support import collect_labeled_image_paths, extract_feature_dataset, label_counts, sample_records_by_label
            from utils.emotion_utils import map_emotion_to_sentiment
            from utils.preprocessors import preprocess_face_image

            face_train_dir = DATASETS_DIR / "face" / "archive" / "train"
            face_test_dir = DATASETS_DIR / "face" / "archive" / "test"
            if not face_train_dir.exists() or not face_test_dir.exists():
                raise FileNotFoundError("Face dataset must exist at NeuroSense/datasets/face/archive/train and test.")

            MAX_TRAIN_PER_SENTIMENT = 1500
            MAX_TEST_PER_SENTIMENT = 500

            train_records = collect_labeled_image_paths(face_train_dir, map_emotion_to_sentiment)
            test_records = collect_labeled_image_paths(face_test_dir, map_emotion_to_sentiment)
            train_records = sample_records_by_label(train_records, per_label=MAX_TRAIN_PER_SENTIMENT, seed=RANDOM_STATE)
            test_records = sample_records_by_label(test_records, per_label=MAX_TEST_PER_SENTIMENT, seed=RANDOM_STATE)

            print("Face train label counts:", label_counts(label for _, label in train_records))
            print("Face test label counts:", label_counts(label for _, label in test_records))

            X_train_raw, y_train_raw = extract_feature_dataset(
                train_records,
                preprocess_face_image,
                cache_path=CACHE_DIR / "face_train_sentiment.npz",
                progress_interval=200,
            )
            X_test_raw, y_test_raw = extract_feature_dataset(
                test_records,
                preprocess_face_image,
                cache_path=CACHE_DIR / "face_test_sentiment.npz",
                progress_interval=200,
            )

            print("Face feature matrix:", X_train_raw.shape, X_test_raw.shape)
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            y_test = le.transform(y_test_raw)

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_raw)
            X_test_sc = scaler.transform(X_test_raw)

            pca = PCA(n_components=100, svd_solver="randomized", random_state=RANDOM_STATE)
            X_train_pca = pca.fit_transform(X_train_sc)
            X_test_pca = pca.transform(X_test_sc)

            model = SVC(kernel="rbf", C=5.0, probability=True, random_state=RANDOM_STATE)
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)

            print(f"Face test accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            print("PCA explained variance:", round(float(np.sum(pca.explained_variance_ratio_)), 4))
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title("Face confusion matrix")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.show()

            figure, axes = plt.subplots(1, 3, figsize=(12, 4))
            for axis, (sample, label) in zip(axes, zip(X_test_raw[:3], y_test_raw[:3])):
                axis.imshow(sample[:2304].reshape(48, 48), cmap="gray")
                axis.set_title(label)
                axis.axis("off")
            plt.tight_layout()
            plt.show()
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            artifact_dir = ARTIFACTS_DIR / "face"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, artifact_dir / "face_model.pkl")
            joblib.dump(scaler, artifact_dir / "face_scaler.pkl")
            joblib.dump(pca, artifact_dir / "face_pca.pkl")
            joblib.dump(le, artifact_dir / "face_label_encoder.pkl")

            print("Saved face artifacts to:", artifact_dir)
            """
        ).strip(),
    ),
]


FUSION_NOTEBOOK = [
    (
        "markdown",
        dedent(
            """
            # 06. Fusion Pipeline

            Build a 12-feature late-fusion dataset from the trained EEG, MEG, speech, and face models, then train a logistic-regression meta-classifier.
            """
        ).strip(),
    ),
    ("code", BOOTSTRAP_CELL),
    (
        "code",
        dedent(
            """
            import joblib
            import numpy as np
            import pandas as pd

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder

            from utils.emotion_utils import SENTIMENT_ORDER, aggregate_probabilities, map_emotion_to_sentiment

            REQUIRED_FILES = [
                ARTIFACTS_DIR / "eeg" / "eeg_model.pkl",
                ARTIFACTS_DIR / "eeg" / "eeg_scaler.pkl",
                ARTIFACTS_DIR / "eeg" / "eeg_label_encoder.pkl",
                ARTIFACTS_DIR / "meg" / "meg_model.pkl",
                ARTIFACTS_DIR / "meg" / "meg_scaler.pkl",
                ARTIFACTS_DIR / "meg" / "meg_label_encoder.pkl",
                ARTIFACTS_DIR / "speech" / "speech_model.pkl",
                ARTIFACTS_DIR / "speech" / "speech_scaler.pkl",
                ARTIFACTS_DIR / "speech" / "speech_label_encoder.pkl",
                ARTIFACTS_DIR / "face" / "face_model.pkl",
                ARTIFACTS_DIR / "face" / "face_scaler.pkl",
                ARTIFACTS_DIR / "face" / "face_pca.pkl",
                ARTIFACTS_DIR / "face" / "face_label_encoder.pkl",
                CACHE_DIR / "speech_features.npz",
                CACHE_DIR / "face_test_sentiment.npz",
            ]

            missing = [str(path) for path in REQUIRED_FILES if not path.exists()]
            if missing:
                raise FileNotFoundError("Run notebooks 01, 02, 04, and 05 before notebook 06. Missing files:\\n" + "\\n".join(missing))
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            eeg_model = joblib.load(ARTIFACTS_DIR / "eeg" / "eeg_model.pkl")
            eeg_scaler = joblib.load(ARTIFACTS_DIR / "eeg" / "eeg_scaler.pkl")
            eeg_encoder = joblib.load(ARTIFACTS_DIR / "eeg" / "eeg_label_encoder.pkl")

            meg_model = joblib.load(ARTIFACTS_DIR / "meg" / "meg_model.pkl")
            meg_scaler = joblib.load(ARTIFACTS_DIR / "meg" / "meg_scaler.pkl")
            meg_encoder = joblib.load(ARTIFACTS_DIR / "meg" / "meg_label_encoder.pkl")

            speech_model = joblib.load(ARTIFACTS_DIR / "speech" / "speech_model.pkl")
            speech_scaler = joblib.load(ARTIFACTS_DIR / "speech" / "speech_scaler.pkl")
            speech_encoder = joblib.load(ARTIFACTS_DIR / "speech" / "speech_label_encoder.pkl")

            face_model = joblib.load(ARTIFACTS_DIR / "face" / "face_model.pkl")
            face_scaler = joblib.load(ARTIFACTS_DIR / "face" / "face_scaler.pkl")
            face_pca = joblib.load(ARTIFACTS_DIR / "face" / "face_pca.pkl")
            face_encoder = joblib.load(ARTIFACTS_DIR / "face" / "face_label_encoder.pkl")

            eeg_df = pd.read_csv(DATASETS_DIR / "eeg" / "eeg" / "emotions.csv")
            eeg_X = eeg_df.drop(columns=["label"]).select_dtypes(include=[np.number]).fillna(0.0).values
            eeg_y = eeg_df["label"].astype(str).values
            _, eeg_X_test, _, eeg_y_test = train_test_split(
                eeg_X,
                eeg_y,
                test_size=0.2,
                stratify=eeg_y,
                random_state=RANDOM_STATE,
            )
            eeg_probs = eeg_model.predict_proba(eeg_scaler.transform(eeg_X_test))

            meg_df = pd.read_csv(DATASETS_DIR / "meg" / "meg_features.csv")
            meg_X = meg_df.drop(columns=["label"]).values
            meg_y = meg_df["label"].astype(str).values
            _, meg_X_test, _, meg_y_test = train_test_split(
                meg_X,
                meg_y,
                test_size=0.2,
                stratify=meg_y,
                random_state=RANDOM_STATE,
            )
            meg_probs = meg_model.predict_proba(meg_scaler.transform(meg_X_test))

            speech_cache = np.load(CACHE_DIR / "speech_features.npz", allow_pickle=True)
            speech_X = speech_cache["X"]
            speech_y = speech_cache["y"].astype(str)
            _, speech_X_test, _, speech_y_test = train_test_split(
                speech_X,
                speech_y,
                test_size=0.2,
                stratify=speech_y,
                random_state=RANDOM_STATE,
            )
            speech_probs = speech_model.predict_proba(speech_scaler.transform(speech_X_test))

            face_cache = np.load(CACHE_DIR / "face_test_sentiment.npz", allow_pickle=True)
            face_X_test = face_cache["X"]
            face_y_test = face_cache["y"].astype(str)
            face_probs = face_model.predict_proba(face_pca.transform(face_scaler.transform(face_X_test)))

            print("EEG test samples:", len(eeg_y_test))
            print("MEG test samples:", len(meg_y_test))
            print("Speech test samples:", len(speech_y_test))
            print("Face test samples:", len(face_y_test))
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            def build_probability_pool(true_labels, probability_matrix, class_labels):
                pool = {sentiment: [] for sentiment in SENTIMENT_ORDER}
                for label, row in zip(true_labels, probability_matrix):
                    sentiment = map_emotion_to_sentiment(label)
                    pooled_row = aggregate_probabilities(class_labels, row, order=SENTIMENT_ORDER)
                    pool[sentiment].append(pooled_row)

                return {
                    sentiment: np.vstack(rows)
                    for sentiment, rows in pool.items()
                    if rows
                }

            modality_pools = {
                "eeg": build_probability_pool(eeg_y_test, eeg_probs, eeg_encoder.classes_),
                "meg": build_probability_pool(meg_y_test, meg_probs, meg_encoder.classes_),
                "speech": build_probability_pool(speech_y_test, speech_probs, speech_encoder.classes_),
                "face": build_probability_pool(face_y_test, face_probs, face_encoder.classes_),
            }

            for modality, pool in modality_pools.items():
                counts = {sentiment: rows.shape[0] for sentiment, rows in pool.items()}
                print(modality, counts)

            for modality, pool in modality_pools.items():
                for sentiment in SENTIMENT_ORDER:
                    if sentiment not in pool:
                        raise ValueError(f"Fusion pool for {modality} is missing sentiment {sentiment}")

            rng = np.random.default_rng(RANDOM_STATE)
            SAMPLES_PER_SENTIMENT = 250
            meta_features = []
            meta_labels = []

            for sentiment in SENTIMENT_ORDER:
                for _ in range(SAMPLES_PER_SENTIMENT):
                    parts = []
                    for modality in ("eeg", "meg", "speech", "face"):
                        rows = modality_pools[modality][sentiment]
                        parts.append(rows[rng.integers(len(rows))])
                    meta_features.append(np.concatenate(parts))
                    meta_labels.append(sentiment)

            X_meta = np.vstack(meta_features).astype(np.float32)
            y_meta_raw = np.asarray(meta_labels)

            fusion_encoder = LabelEncoder()
            y_meta = fusion_encoder.fit_transform(y_meta_raw)

            X_train, X_test, y_train, y_test = train_test_split(
                X_meta,
                y_meta,
                test_size=0.2,
                stratify=y_meta,
                random_state=RANDOM_STATE,
            )

            fusion_model = LogisticRegression(max_iter=2000, multi_class="auto", random_state=RANDOM_STATE)
            fusion_model.fit(X_train, y_train)
            y_pred = fusion_model.predict(X_test)

            print("Fusion feature matrix:", X_meta.shape)
            print(f"Fusion test accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(classification_report(y_test, y_pred, target_names=fusion_encoder.classes_))
            """
        ).strip(),
    ),
    (
        "code",
        dedent(
            """
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="crest",
                xticklabels=fusion_encoder.classes_,
                yticklabels=fusion_encoder.classes_,
            )
            plt.title("Fusion confusion matrix")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.show()

            artifact_dir = ARTIFACTS_DIR / "fusion"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(fusion_model, artifact_dir / "fusion_model.pkl")
            joblib.dump(fusion_encoder, artifact_dir / "fusion_label_encoder.pkl")

            print("Saved fusion artifacts to:", artifact_dir)
            """
        ).strip(),
    ),
]


NOTEBOOK_SPECS = {
    "01_EEG_Emotion_Recognition.ipynb": EEG_NOTEBOOK,
    "02_MEG_Emotion_Recognition.ipynb": MEG_NOTEBOOK,
    "03_MRI_Brain_Tumor.ipynb": MRI_NOTEBOOK,
    "04_Speech_Emotion_Recognition.ipynb": SPEECH_NOTEBOOK,
    "05_Face_Emotion_Recognition.ipynb": FACE_NOTEBOOK,
    "06_Fusion_Pipeline.ipynb": FUSION_NOTEBOOK,
}


for notebook_name, cells in NOTEBOOK_SPECS.items():
    write_notebook(notebook_name, cells)

print("Regenerated all NeuroSense notebooks.")
