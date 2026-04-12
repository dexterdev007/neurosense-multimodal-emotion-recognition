"""
Shared preprocessing functions used by both training notebooks and the FastAPI backend.

Important rule:
These functions must stay numerically aligned with the notebooks. Otherwise the backend
would feed the model data that is shaped differently from the data seen during training.
"""

from __future__ import annotations

import io
import warnings

import cv2
import numpy as np


MRI_IMAGE_SIZE = (64, 64)
FACE_IMAGE_SIZE = (48, 48)
SPEECH_FEATURE_COUNT = 162
SPEECH_PADDING_LENGTH = 12


def _as_row_vector(values: list[float] | np.ndarray, dtype: np.dtype = np.float64) -> np.ndarray:
    """Return a single-sample matrix with shape `(1, n_features)`."""
    return np.asarray(values, dtype=dtype).reshape(1, -1)


def _decode_grayscale_image(image_bytes: bytes, error_message: str) -> np.ndarray:
    """Decode raw image bytes into a grayscale OpenCV image."""
    buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(error_message)
    return image


def preprocess_eeg(features_list: list[float]) -> np.ndarray:
    """Convert a raw EEG feature list into the row-vector shape expected by the scaler."""
    return _as_row_vector(features_list)


def preprocess_meg(features_list: list[float]) -> np.ndarray:
    """Convert a raw MEG feature list into the row-vector shape expected by the scaler."""
    return _as_row_vector(features_list)


def preprocess_mri_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert an MRI image into the flat feature vector used during training.

    Steps:
    1. Decode the uploaded image.
    2. Convert it to grayscale.
    3. Resize it to 64 x 64.
    4. Flatten all pixels into one long vector.
    """
    image = _decode_grayscale_image(
        image_bytes,
        "Could not decode image. Ensure it is a valid JPEG or PNG.",
    )
    resized_image = cv2.resize(image, MRI_IMAGE_SIZE)
    flat_pixels = resized_image.flatten().astype(np.float64)
    return _as_row_vector(flat_pixels)


def preprocess_speech(audio_bytes: bytes) -> np.ndarray:
    """
    Extract the fixed-length speech feature vector used by the speech notebook.

    The output contains exactly 162 values. They combine several common audio
    descriptors so a classical model like SVM can work on voice data:

    - MFCC mean and standard deviation
    - chroma mean and standard deviation
    - mel-spectrogram summaries
    - spectral contrast
    - tonnetz
    - simple signal statistics such as ZCR, RMS, and rolloff
    - zero-padding at the end so the feature size stays constant
    """
    import librosa

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"n_fft=.* is too large for input signal of length=.*",
            category=UserWarning,
        )

        waveform, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=4.0)

        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate, n_chroma=12)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=20)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)[:10]

        contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate)
        contrast_mean = np.mean(contrast, axis=1)

        harmonic = librosa.effects.harmonic(waveform)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sample_rate)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(waveform)))
        root_mean_square = float(np.mean(librosa.feature.rms(y=waveform)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)))

    padding = np.zeros(SPEECH_PADDING_LENGTH)

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            chroma_mean,
            chroma_std,
            mel_mean,
            mel_std,
            contrast_mean,
            tonnetz_mean,
            [zero_crossing_rate, root_mean_square, spectral_rolloff],
            padding,
        ]
    )

    assert len(features) == SPEECH_FEATURE_COUNT, f"Feature count: {len(features)}"
    return _as_row_vector(features)


def preprocess_face_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert a face image into the exact feature vector used by the face notebook.

    Steps:
    1. Decode image bytes.
    2. Convert to grayscale.
    3. Resize to 48 x 48.
    4. Flatten and normalize pixel values.
    5. Append four simple summary statistics.
    """
    image = _decode_grayscale_image(image_bytes, "Could not decode face image.")
    resized_image = cv2.resize(image, FACE_IMAGE_SIZE)

    pixels = resized_image.flatten().astype(np.float32)
    normalized_pixels = pixels / 255.0
    statistics = np.array(
        [
            np.mean(pixels),
            np.std(pixels),
            np.percentile(pixels, 25),
            np.percentile(pixels, 75),
        ]
    )
    features = np.concatenate([normalized_pixels, statistics])
    return _as_row_vector(features)  # shape (1, 2308)
