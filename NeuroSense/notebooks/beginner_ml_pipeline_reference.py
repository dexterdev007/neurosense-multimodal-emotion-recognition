from __future__ import annotations

"""
Beginner-friendly reference for the NeuroSense machine-learning pipeline.

Purpose:
- explain the repeated training pattern used across the notebooks
- show the main scikit-learn steps in small functions
- give one place where a student can understand the project without opening every notebook first

This file is educational. The real training happens inside the numbered notebooks.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModalitySummary:
    name: str
    input_type: str
    preprocessing: str
    model: str
    notebook: str


MODALITIES = [
    ModalitySummary(
        name="EEG",
        input_type="CSV rows of numeric brain-signal features",
        preprocessing="fill missing values, label encoding, train/test split, standard scaling",
        model="SVM with RBF kernel",
        notebook="01_EEG_Emotion_Recognition.ipynb",
    ),
    ModalitySummary(
        name="MEG",
        input_type="CSV rows of MEG features",
        preprocessing="same as EEG, with synthetic fallback if a real file is missing",
        model="SVM with RBF kernel",
        notebook="02_MEG_Emotion_Recognition.ipynb",
    ),
    ModalitySummary(
        name="MRI",
        input_type="brain scan images",
        preprocessing="grayscale conversion, resize to 64x64, flatten, scale, PCA",
        model="SVM with RBF kernel",
        notebook="03_MRI_Brain_Tumor.ipynb",
    ),
    ModalitySummary(
        name="Speech",
        input_type="audio recordings",
        preprocessing="extract 162 audio features, actor-aware evaluation (RAVDESS), scaling",
        model="SVM with actor-holdout evaluation",
        notebook="train_speech_ravdess.py",
    ),
    ModalitySummary(
        name="Face",
        input_type="face images",
        preprocessing="grayscale conversion, resize to 48x48, flatten, simple statistics, scale, PCA",
        model="SVM with fixed PCA + SVM settings",
        notebook="05_Face_Emotion_Recognition.ipynb",
    ),
    ModalitySummary(
        name="Fusion",
        input_type="probability outputs from the base modalities",
        preprocessing="stack one 3-class probability vector per modality into one feature vector",
        model="LogisticRegression meta-model",
        notebook="06_Fusion_Pipeline.ipynb",
    ),
]


def encode_labels(raw_labels: Sequence[str]) -> tuple[LabelEncoder, np.ndarray]:
    """
    Convert text labels into integer IDs.

    Example:
    ['NEGATIVE', 'POSITIVE', 'NEUTRAL'] -> [0, 2, 1]
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(raw_labels)
    return encoder, encoded_labels


def split_dataset(
    feature_matrix: np.ndarray,
    encoded_labels: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into a training part and a test part.

    Why this matters:
    the model should be evaluated on unseen examples, not on the same rows it learned from.
    """
    return train_test_split(
        feature_matrix,
        encoded_labels,
        test_size=test_size,
        stratify=encoded_labels,
        random_state=random_state,
    )


def scale_train_and_test_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """
    Standardize features so every column is on a similar scale.

    This especially helps SVM because distance and margin calculations become more stable.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    return scaler, train_scaled, test_scaled


def apply_pca_when_needed(
    train_features: np.ndarray,
    test_features: np.ndarray,
    n_components: int,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """
    Reduce very large feature vectors into a smaller set of informative components.

    MRI and face data use this because images contain thousands of pixel values.
    """
    pca = PCA(n_components=n_components)
    train_reduced = pca.fit_transform(train_features)
    test_reduced = pca.transform(test_features)
    return pca, train_reduced, test_reduced


def train_svm_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    c_value: float = 5.0,
    random_state: int = 42,
) -> SVC:
    """Train the classical classifier used in most base NeuroSense modules."""
    model = SVC(kernel="rbf", C=c_value, probability=True, random_state=random_state)
    model.fit(train_features, train_labels)
    return model


def evaluate_classifier(
    model: SVC | LogisticRegression,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, str | float]:
    """
    Return the simplest two outputs a beginner usually wants first:
    accuracy and a full per-class classification report.
    """
    predictions = model.predict(test_features)
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "classification_report": classification_report(test_labels, predictions, target_names=list(class_names)),
    }


def build_fusion_feature_matrix(modality_probability_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """
    Stack multiple probability vectors side by side.

    If four modalities each output three probabilities, the fusion row becomes:
    [eeg_3 values | meg_3 values | speech_3 values | face_3 values]
    """
    return np.concatenate([np.asarray(vector, dtype=np.float64).reshape(-1) for vector in modality_probability_vectors])


def train_fusion_model(
    fusion_train_features: np.ndarray,
    fusion_train_labels: np.ndarray,
    *,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train the late-fusion model that learns from the confidence scores of other models.
    """
    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(fusion_train_features, fusion_train_labels)
    return model


def print_project_summary() -> None:
    """Print a readable summary of the full project for quick revision."""
    print("NeuroSense Beginner Pipeline Summary")
    print("=" * 40)
    for modality in MODALITIES:
        print(f"\n{modality.name}")
        print(f"  Input:         {modality.input_type}")
        print(f"  Preprocessing: {modality.preprocessing}")
        print(f"  Model:         {modality.model}")
        print(f"  Notebook:      {modality.notebook}")

    print("\nCommon base-model flow")
    print("  1. Read data or extract features")
    print("  2. Build X (features) and y (labels)")
    print("  3. Encode labels to integers")
    print("  4. Split into train and test sets")
    print("  5. Scale features")
    print("  6. Optionally apply PCA for image-heavy data")
    print("  7. Train a classifier")
    print("  8. Evaluate on unseen test data")
    print("  9. Save artifacts for the backend")

    print("\nWhy fusion exists")
    print("  Each base model sees only one modality.")
    print("  The fusion model sees the confidence scores from several modalities together.")
    print("  That lets the final prediction use more than one source of evidence.")


if __name__ == "__main__":
    print_project_summary()
