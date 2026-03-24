"""
CRITICAL: These functions MUST exactly match the preprocessing in training notebooks.
Any change here must also be reflected in the corresponding notebook.
"""
import numpy as np
import cv2
import io

def preprocess_eeg(features_list: list) -> np.ndarray:
    """Accepts raw list of floats, returns shaped array for scaler.transform()"""
    return np.array(features_list, dtype=np.float64).reshape(1, -1)

def preprocess_meg(features_list: list) -> np.ndarray:
    return np.array(features_list, dtype=np.float64).reshape(1, -1)

def preprocess_mri_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocessing MUST match training notebook exactly:
    1. Decode image bytes
    2. Convert to grayscale
    3. Resize to 64x64
    4. Flatten to 4096-dim vector
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPEG or PNG.")
    img = cv2.resize(img, (64, 64))
    return img.flatten().reshape(1, -1).astype(np.float64)

def preprocess_speech(audio_bytes: bytes) -> np.ndarray:
    """
    Extracts exactly 162 features — MUST match notebook extract_speech_features().
    Layout:
      [0:40]    MFCC means        (40)
      [40:80]   MFCC stds         (40)
      [80:92]   Chroma means      (12)
      [92:104]  Chroma stds       (12)
      [104:124] Mel means         (20)
      [124:134] Mel stds first10  (10)
      [134:141] Spectral contrast (7)
      [141:147] Tonnetz           (6)
      [147:150] ZCR, RMS, Rolloff (3)
      [150:162] Zero padding      (12)
      TOTAL = 162
    """
    import librosa

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=4.0)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std  = np.std(chroma, axis=1)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel_mean = np.mean(mel, axis=1)
    mel_std  = np.std(mel, axis=1)[:10]

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    harmonic = librosa.effects.harmonic(y)
    tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)

    zcr     = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms     = float(np.mean(librosa.feature.rms(y=y)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    padding = np.zeros(12)

    features = np.concatenate([
        mfcc_mean, mfcc_std,
        chroma_mean, chroma_std,
        mel_mean, mel_std,
        contrast_mean,
        tonnetz_mean,
        [zcr, rms, rolloff],
        padding,
    ])

    assert len(features) == 162, f"Feature count: {len(features)}"
    return features.reshape(1, -1)

def preprocess_face_image(image_bytes: bytes) -> np.ndarray:
    """
    MUST match notebook extract_face_features():
    1. Decode image
    2. Convert to grayscale
    3. Resize to 48x48
    4. Flatten pixels / 255.0
    5. Add 4 statistical features
    Total: 2308 features
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode face image.")
    img = cv2.resize(img, (48, 48))
    pixels = img.flatten().astype(np.float32)
    norm_pixels = pixels / 255.0
    stats = np.array([
        np.mean(pixels), np.std(pixels),
        np.percentile(pixels, 25), np.percentile(pixels, 75)
    ])
    features = np.concatenate([norm_pixels, stats])
    return features.reshape(1, -1)  # shape (1, 2308)
