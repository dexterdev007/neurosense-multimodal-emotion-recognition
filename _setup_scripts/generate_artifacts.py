import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

os.makedirs('NeuroSense/artifacts/eeg', exist_ok=True)
os.makedirs('NeuroSense/artifacts/meg', exist_ok=True)
os.makedirs('NeuroSense/artifacts/mri', exist_ok=True)
os.makedirs('NeuroSense/artifacts/speech', exist_ok=True)
os.makedirs('NeuroSense/artifacts/face', exist_ok=True)
os.makedirs('NeuroSense/artifacts/fusion', exist_ok=True)

# Common labels
labels_3 = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

# EEG (32 features)
X_eeg = np.random.randn(100, 32)
y_eeg = np.random.choice(labels_3, 100)
scaler_eeg = StandardScaler().fit(X_eeg)
le_eeg = LabelEncoder().fit(y_eeg)
model_eeg = SVC(kernel='rbf', probability=True).fit(scaler_eeg.transform(X_eeg), le_eeg.transform(y_eeg))
joblib.dump(model_eeg, 'NeuroSense/artifacts/eeg/eeg_model.pkl')
joblib.dump(scaler_eeg, 'NeuroSense/artifacts/eeg/eeg_scaler.pkl')
joblib.dump(le_eeg, 'NeuroSense/artifacts/eeg/eeg_label_encoder.pkl')

# MEG (50 features)
X_meg = np.random.randn(100, 50)
y_meg = np.random.choice(labels_3, 100)
scaler_meg = StandardScaler().fit(X_meg)
le_meg = LabelEncoder().fit(y_meg)
model_meg = SVC(kernel='rbf', probability=True).fit(scaler_meg.transform(X_meg), le_meg.transform(y_meg))
joblib.dump(model_meg, 'NeuroSense/artifacts/meg/meg_model.pkl')
joblib.dump(scaler_meg, 'NeuroSense/artifacts/meg/meg_scaler.pkl')
joblib.dump(le_meg, 'NeuroSense/artifacts/meg/meg_label_encoder.pkl')

# MRI (4096 features -> 150 PCA)
labels_mri = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
X_mri = np.random.rand(100, 4096)
y_mri = np.random.choice(labels_mri, 100)
scaler_mri = StandardScaler().fit(X_mri)
pca_mri = PCA(n_components=min(100, 150)).fit(scaler_mri.transform(X_mri)) # n_samples = 100, max components is 100
le_mri = LabelEncoder().fit(y_mri)
model_mri = SVC(kernel='rbf', probability=True).fit(pca_mri.transform(scaler_mri.transform(X_mri)), le_mri.transform(y_mri))
joblib.dump(model_mri, 'NeuroSense/artifacts/mri/mri_model.pkl')
joblib.dump(scaler_mri, 'NeuroSense/artifacts/mri/mri_scaler.pkl')
joblib.dump(pca_mri, 'NeuroSense/artifacts/mri/mri_pca.pkl')
joblib.dump(le_mri, 'NeuroSense/artifacts/mri/mri_label_encoder.pkl')

# Speech (41 features)
labels_speech = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
X_speech = np.random.randn(100, 41)
y_speech = np.random.choice(labels_speech, 100)
scaler_speech = StandardScaler().fit(X_speech)
le_speech = LabelEncoder().fit(y_speech)
model_speech = SVC(kernel='rbf', probability=True).fit(scaler_speech.transform(X_speech), le_speech.transform(y_speech))
joblib.dump(model_speech, 'NeuroSense/artifacts/speech/speech_model.pkl')
joblib.dump(scaler_speech, 'NeuroSense/artifacts/speech/speech_scaler.pkl')
joblib.dump(le_speech, 'NeuroSense/artifacts/speech/speech_label_encoder.pkl')

# Face (2308 features -> 100 PCA)
labels_face = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
X_face = np.random.rand(150, 2308)
y_face = np.random.choice(labels_face, 150)
scaler_face = StandardScaler().fit(X_face)
pca_face = PCA(n_components=100).fit(scaler_face.transform(X_face))
le_face = LabelEncoder().fit(y_face)
model_face = SVC(kernel='rbf', probability=True).fit(pca_face.transform(scaler_face.transform(X_face)), le_face.transform(y_face))
joblib.dump(model_face, 'NeuroSense/artifacts/face/face_model.pkl')
joblib.dump(scaler_face, 'NeuroSense/artifacts/face/face_scaler.pkl')
joblib.dump(pca_face, 'NeuroSense/artifacts/face/face_pca.pkl')
joblib.dump(le_face, 'NeuroSense/artifacts/face/face_label_encoder.pkl')

# Fusion (12 features)
X_fusion = np.random.rand(100, 12)
y_fusion = np.random.choice(labels_3, 100)
le_fusion = LabelEncoder().fit(y_fusion)
model_fusion = LogisticRegression().fit(X_fusion, le_fusion.transform(y_fusion))
joblib.dump(model_fusion, 'NeuroSense/artifacts/fusion/fusion_model.pkl')
joblib.dump(le_fusion, 'NeuroSense/artifacts/fusion/fusion_label_encoder.pkl')

print("All ML artifacts generated successfully!")
