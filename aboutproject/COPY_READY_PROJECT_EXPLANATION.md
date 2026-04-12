# NeuroSense: Copy-Ready Project Explanation

## 1. Introduction

NeuroSense is a multimodal machine learning project designed to study human emotion and related human state signals using different types of input data. The system combines five modalities:

- EEG
- MEG
- MRI
- Speech
- Face

Each modality is processed by its own machine learning pipeline. After the individual models are trained, their outputs are combined through a final fusion stage. This makes the project a complete end-to-end ML system, because it covers:

- data handling
- preprocessing
- feature extraction
- model training
- evaluation
- artifact saving
- backend inference
- frontend interaction

The project is intentionally built in a way that is easy to explain academically. Instead of using deep and highly complex architectures everywhere, the project mainly uses classical machine learning methods such as SVM, PCA, scaling, and Logistic Regression. This makes the project suitable for presentation, viva, and technical understanding.

## 2. Main Aim Of The Project

The main aim of NeuroSense is to build a system that can analyze human-related signals from multiple modalities and produce a prediction from each one individually, and also a final combined prediction through fusion.

This project is not only about obtaining a prediction. It is also about demonstrating the complete ML workflow:

1. collecting data
2. converting raw data into useful features
3. training models
4. evaluating results
5. saving trained artifacts
6. serving predictions through an API
7. integrating everything into a usable interface

So this project can be understood as both:

- a multimodal emotion-related recognition system
- a full-stack machine learning deployment project

## 3. Core Idea Of Multimodal Learning

Multimodal learning means using more than one type of input to understand a problem better.

In many real systems, one single modality may be noisy or incomplete. For example:

- facial expression may not always reflect the real internal emotion
- speech may be affected by accent, speaker style, or noise
- EEG may contain noise or limited contextual information

By using multiple modalities, the system can collect evidence from different sources. That is the reason a fusion model is added at the end.

In NeuroSense:

- EEG and MEG represent brain-signal-style structured features
- MRI represents medical image data
- speech represents audio-based emotion cues
- face represents visual facial cues

These models work independently first, then their outputs are combined.

## 4. Full Project Workflow

The complete working of the project can be explained in two phases.

### 4.1 Training Phase

In the training phase:

1. Each notebook loads a dataset.
2. Raw samples are converted into numeric features.
3. Labels are encoded.
4. Data is split into training and testing sets or uses a predefined split.
5. Features are scaled.
6. PCA is applied where needed.
7. A classifier is trained.
8. Performance is evaluated.
9. Trained objects are saved as artifacts.

### 4.2 Inference Phase

In the inference phase:

1. The backend loads the saved model files.
2. A user sends input through the API or frontend.
3. The backend applies the same preprocessing as training.
4. The saved model predicts the class.
5. The backend returns:
   - prediction
   - confidence
   - class probabilities

For fusion:

1. multiple modality outputs are collected
2. each output is converted into a common sentiment space
3. the final fusion logic combines them
4. one final result is returned

## 5. Project Architecture

The project has three major parts:

### 5.1 Training Notebooks

The notebooks are used for model development and artifact generation.

- `01_EEG_Emotion_Recognition.ipynb`
- `02_MEG_Emotion_Recognition.ipynb`
- `03_MRI_Brain_Tumor.ipynb`
- Speech (RAVDESS): `NeuroSense/notebooks/train_speech_ravdess.py`
- `05_Face_Emotion_Recognition.ipynb`
- `06_Fusion_Pipeline.ipynb`

### 5.2 Artifacts

After training, each notebook saves files such as:

- model
- scaler
- PCA object
- label encoder
- metadata

These are stored under `NeuroSense/artifacts/`.

### 5.3 Web Application

The web application has:

- backend in `NeuroSense/webdev/backend/`
- frontend in `NeuroSense/webdev/frontend/`

The backend is built using FastAPI.  
The frontend provides an interface for input and result display.

## 6. Folder Structure Meaning

Important folders are:

- `NeuroSense/notebooks/`
  Contains all ML training notebooks.

- `NeuroSense/artifacts/`
  Contains trained ML artifacts like `.pkl`, metadata, and cached features.

- `NeuroSense/webdev/backend/`
  Contains FastAPI routes, preprocessing logic, model loading logic, and metadata loading.

- `NeuroSense/webdev/frontend/`
  Contains HTML, CSS, and JavaScript for the user-facing interface.

## 7. Common Machine Learning Theory Used In This Project

### 7.1 Supervised Learning

This project is based mainly on supervised learning.

In supervised learning:

- inputs are already labeled
- the model learns a mapping from input to output

Examples:

- EEG features -> emotion label
- face image -> sentiment class
- speech features -> emotion class

### 7.2 Classification

The project mainly solves classification problems.

Classification means the output belongs to a fixed category.

Examples:

- `NEGATIVE`
- `NEUTRAL`
- `POSITIVE`
- MRI tumor classes like `glioma`, `meningioma`, `pituitary`, `no_tumor`

### 7.3 Features And Labels

In ML, two key terms are:

- `X` = features
- `y` = labels

Features are the input information used for learning.  
Labels are the correct target outputs.

Examples:

- EEG CSV columns are features
- image pixel values become features after preprocessing
- speech descriptors like MFCC become features

### 7.4 Train-Test Split

To measure generalization, the model should be tested on unseen data.

That is why the project uses:

- training set
- test set

Training set is used to learn patterns.  
Test set is used to check how well the model performs on unseen samples.

### 7.5 Label Encoding

Machine learning models usually need numeric output labels.

So text labels like:

- `"happy"`
- `"sad"`
- `"NEGATIVE"`

are converted into integers by `LabelEncoder`.

Later, the label encoder is also used to convert predictions back into readable text.

### 7.6 Feature Scaling

Many classical ML models work better when features are on a similar numeric scale.

This is why `StandardScaler` is used.

It performs:

- mean subtraction
- standard deviation normalization

This is important especially for SVM.

### 7.7 SVM

The base models in this project mainly use Support Vector Machine with RBF kernel.

Why SVM is used:

- strong classical baseline
- works well on structured numeric features
- easier to explain than deep neural networks
- suitable for academic projects

RBF kernel helps the model learn non-linear decision boundaries, which is useful when simple linear separation is not enough.

### 7.8 PCA

PCA means Principal Component Analysis.

PCA is used in image-based pipelines such as MRI and face.

Why PCA is needed:

- image vectors are very large
- many dimensions are redundant
- smaller feature spaces help classical models work better

PCA reduces dimensionality while preserving much of the useful variance.

### 7.9 Logistic Regression

The fusion stage uses Logistic Regression.

Why it is used:

- fusion input is already a compact probability vector
- Logistic Regression is simple and interpretable
- it works well for combining structured probability information

### 7.10 Overfitting

Overfitting means the model learns the training data too specifically and fails to generalize well.

A common sign is:

- high training accuracy
- much lower testing accuracy

This project checks evaluation metrics so that overfitting can be observed and discussed honestly.

### 7.11 Data Leakage

Data leakage happens when information from the test side leaks into training, causing overly optimistic results.

This is especially important in speech emotion recognition.

If the same speaker appears in both training and testing, the model may partially learn speaker identity instead of emotion.

That is why the speech module uses actor-holdout and GroupKFold evaluation on RAVDESS (speaker-aware splits).

## 8. Common ML Pipeline Used Across The Project

Although the data types change, the overall machine learning flow is similar in most notebooks.

The general pipeline is:

1. load data
2. prepare features `X`
3. prepare labels `y`
4. encode labels
5. split train and test
6. scale features
7. apply PCA if needed
8. train classifier
9. evaluate model
10. save artifacts

A common code pattern looks like this:

```python
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

Meaning:

- `LabelEncoder()` converts class names into numbers
- `train_test_split()` creates a fair test set
- `StandardScaler()` normalizes feature ranges
- `model.fit()` trains the model
- `model.predict()` generates predictions

## 9. Modality-Wise Full Explanation

### 9.1 EEG Modality

#### What EEG Means

EEG stands for Electroencephalography.  
It measures electrical activity from the brain using scalp electrodes.

In this project, EEG data is already available in tabular CSV form.

#### Why EEG Is Simpler In This Project

Because the EEG dataset is already numeric, it does not need image decoding or audio feature extraction.

#### EEG Pipeline

1. load EEG CSV
2. separate feature columns and label column
3. fill missing values if needed
4. encode labels
5. split into train and test
6. scale features
7. train SVM
8. evaluate
9. save model files

#### Why SVM Works Here

EEG in this project is already a structured numeric feature matrix, so SVM is a natural classical ML choice.

#### EEG Artifacts

- `eeg_model.pkl`
- `eeg_scaler.pkl`
- `eeg_label_encoder.pkl`

### 9.2 MEG Modality

#### What MEG Means

MEG stands for Magnetoencephalography.  
It measures magnetic fields generated by neural activity.

#### Important Project Reality

Real MEG data is difficult to collect in a student environment because it requires expensive and specialized hardware.  
So in this project, if a real MEG CSV is missing, a synthetic MEG-like dataset is created to demonstrate the pipeline.

#### MEG Pipeline

1. check whether a MEG CSV exists
2. if not, create synthetic data
3. separate features and labels
4. encode labels
5. split into train and test
6. scale features
7. train SVM
8. evaluate
9. save artifacts and metadata

#### Why This Is Still Useful

Even though the MEG part may use synthetic data, it still demonstrates:

- dataset preparation
- ML training flow
- artifact creation
- backend integration

So it is useful as a technical pipeline demo, but it should not be presented as real clinical MEG performance.

#### MEG Artifacts

- `meg_model.pkl`
- `meg_scaler.pkl`
- `meg_label_encoder.pkl`

### 9.3 MRI Modality

#### What MRI Means

MRI stands for Magnetic Resonance Imaging.  
In this project, the MRI notebook performs brain tumor classification from MRI images.

#### Problem With Classical ML On Images

Classical models like SVM cannot directly use raw image files.

So each image must first be converted into a numeric vector.

#### MRI Pipeline

1. collect image paths from training and testing folders
2. read image bytes
3. convert image to grayscale
4. resize to `64 x 64`
5. flatten image into a 1D vector
6. encode class labels
7. scale the features
8. apply PCA
9. train SVM
10. evaluate on test images
11. save model, scaler, PCA, and encoder

#### Why Grayscale Is Used

MRI does not need color information for this task.  
Grayscale reduces complexity.

#### Why PCA Is Used

`64 x 64` images create 4096 raw pixel features.  
PCA reduces this to a smaller feature space while keeping major information.

#### MRI Artifacts

- `mri_model.pkl`
- `mri_scaler.pkl`
- `mri_pca.pkl`
- `mri_label_encoder.pkl`

### 9.4 Speech Modality

#### What The Speech Module Does

The speech module predicts sentiment from audio recordings.

#### Why Raw Audio Is Not Used Directly

Classical ML models cannot directly use raw audio waveforms in a simple tabular way.  
So raw sound must be converted into a fixed-length feature vector.

#### Speech Feature Extraction Theory

The speech preprocessing function extracts 162 numeric values using audio descriptors such as:

- MFCC means
- MFCC standard deviations
- chroma features
- mel-spectrogram features
- spectral contrast
- tonnetz
- zero crossing rate
- RMS energy
- spectral rolloff

These features summarize the frequency and energy behavior of speech.

#### Speech Pipeline

1. collect RAVDESS speech audio paths (Actor_01 to Actor_24)
2. infer labels from the RAVDESS filename emotion code
3. map emotions into 3 sentiment classes (`NEGATIVE`, `NEUTRAL`, `POSITIVE`)
4. extract fixed-length features from each file (162 values)
5. encode labels
6. split data by actor (actor-holdout)
7. scale features
8. train SVM
9. evaluate actor-holdout accuracy + GroupKFold cross-validation
10. save artifacts + metadata

#### Why Actor-Holdout Is Important

If the same speaker appears in both train and test, the model may memorize speaker traits instead of emotional patterns.

Actor-holdout is more honest because:

- the model trains on some actors
- the model is tested on unseen actors

This gives a better estimate of real generalization.

#### Speech Artifacts

- `speech_model.pkl`
- `speech_scaler.pkl`
- `speech_label_encoder.pkl`

### 9.5 Face Modality

#### What The Face Module Does

The face module predicts sentiment from facial images.

#### Why Face Is An Image Problem

Like MRI, face data is image-based.  
So raw images must be converted into numeric vectors before SVM can use them.

#### Face Pipeline

1. collect train and test image paths
2. map detailed face emotions into three sentiment classes
3. decode image
4. convert to grayscale
5. resize to `48 x 48`
6. flatten image pixels
7. normalize pixel values
8. add simple summary statistics
9. encode labels
10. scale features
11. apply PCA
12. train SVM
13. evaluate
14. save artifacts

#### Why Emotion Mapping Is Used

The original dataset may contain many detailed emotions, but the project simplifies them into:

- `NEGATIVE`
- `NEUTRAL`
- `POSITIVE`

This makes fusion easier, because multiple modalities can share the same sentiment space.

#### Why Face Would Benefit From CNN

In image tasks, CNN is usually better than flattening pixels and using SVM.  
CNN preserves spatial structure and learns patterns such as:

- eyebrows
- mouth shape
- eye openness
- local facial structure

However, in this project, classical ML is used for simplicity and explainability.

#### Face Artifacts

- `face_model.pkl`
- `face_scaler.pkl`
- `face_pca.pkl`
- `face_label_encoder.pkl`

### 9.6 Fusion Modality

#### What Fusion Means

Fusion combines outputs from multiple base models into one final decision.

#### Why Fusion Is Needed

Each modality sees only one part of the problem:

- face sees visual expression
- speech sees vocal emotion cues
- EEG/MEG see structured signal patterns
- MRI sees image structure

Fusion allows the system to use combined evidence.

#### Type Of Fusion Used

This project uses late fusion.

Late fusion means:

- each modality first predicts independently
- their output probabilities are then combined

#### Fusion Pipeline

1. load trained base models
2. collect probability outputs from EEG, MEG, speech, and face
3. convert all outputs into a common three-class sentiment format
4. concatenate those probabilities
5. build a new fusion dataset
6. train Logistic Regression as the fusion model
7. compare fusion with:
   - simple average
   - majority vote
8. save fusion model and metadata

#### Why Common Sentiment Space Is Needed

Different modalities may use different original labels.  
For fusion to work properly, they are converted into the same shared sentiment categories:

- `NEGATIVE`
- `NEUTRAL`
- `POSITIVE`

#### Why Logistic Regression Is Used

The fusion input is already a clean structured probability vector.  
Logistic Regression is sufficient, simple, and interpretable for that space.

#### Fusion Artifacts

- `fusion_model.pkl`
- `fusion_label_encoder.pkl`

## 10. Shared Helper Code And Why It Matters

### 10.1 `notebook_support.py`

This file provides shared notebook helper functions, such as:

- locating the project root
- setting paths
- collecting image paths
- collecting audio paths
- converting many files into `X` and `y`

This avoids repeating the same support code in every notebook.

### 10.2 `preprocessors.py`

This is one of the most important technical files.

It contains the preprocessing functions used by both:

- training notebooks
- backend inference routes

Examples:

- `preprocess_eeg()`
- `preprocess_meg()`
- `preprocess_mri_image()`
- `preprocess_speech()`
- `preprocess_face_image()`

This ensures training-time and inference-time preprocessing stay aligned.

### 10.3 `model_loader.py`

This file loads all saved artifacts at backend startup.

Why it matters:

- avoids reloading models on every request
- keeps inference efficient
- centralizes artifact access

### 10.4 `metadata_loader.py`

This file loads modality metadata, such as:

- data source
- evaluation method
- project honesty notes

This metadata is useful for transparency and for displaying honest caveats in the UI.

### 10.5 `emotion_utils.py`

This file contains logic for mapping different detailed labels into a common sentiment representation.

This is especially useful in fusion.

## 11. Why Saved Artifacts Are Necessary

When a model is trained, the project saves more than only the classifier.

These objects usually include:

- model
- scaler
- PCA object
- label encoder
- metadata

Each one has an important role:

- **model** stores learned decision boundaries
- **scaler** stores exact normalization parameters
- **PCA** stores exact dimensionality reduction mapping
- **label encoder** stores class-to-index mapping
- **metadata** stores transparency information

Without these saved artifacts, the backend could not reproduce the same behavior seen during training.

## 12. Backend Working

The backend is built using FastAPI.

Important backend files:

- `app.py`
- `utils/model_loader.py`
- `utils/preprocessors.py`
- router files for EEG, MEG, MRI, speech, face, fusion

### Backend Flow

1. FastAPI starts
2. all trained artifacts are loaded
3. user sends an input request
4. the correct route is selected
5. the route calls the correct preprocessing function
6. processed data is passed into scaler, PCA, and model
7. prediction and confidence are returned

### Example

For face:

1. user uploads image
2. backend reads bytes
3. `preprocess_face_image()` converts it to feature vector
4. scaler normalizes the features
5. PCA reduces dimensions
6. SVM predicts the class
7. JSON response is returned

## 13. Frontend Working

The frontend provides the user interface for testing the system.

It allows:

- entering numeric values
- uploading images
- uploading audio
- triggering combined prediction
- showing final result and modality contribution

The frontend and backend are connected through HTTP API calls.

## 14. Evaluation And Honesty

This project includes honesty and transparency as an important principle.

Examples:

- MEG clearly indicates when synthetic data is used
- speech distinguishes honest speaker-holdout accuracy from optimistic random-split accuracy
- fusion includes metadata about whether the trained meta-model actually beats a simple baseline

This is important in academic presentation because it shows technical maturity.

## 15. Strengths Of The Project

- covers full ML lifecycle
- includes multiple modalities
- uses clear classical ML pipelines
- includes artifact saving and deployment
- supports backend and frontend integration
- uses transparent metadata
- includes a fusion stage

## 16. Limitations Of The Project

- face would likely perform better with CNN
- MEG may be synthetic and should not be overclaimed
- speech is challenging because of speaker variation
- classical image pipelines are weaker than deep learning image pipelines
- multimodal fusion quality depends on the quality of the base models

## 17. Future Improvements

Possible future work includes:

- use CNN for face emotion recognition
- use spectrogram-based CNN for speech
- use stronger EEG and MEG datasets
- improve fusion with better modality confidence calibration
- add subject-wise or speaker-wise more advanced evaluation
- improve deployment and live inference capabilities

## 18. Short Viva Summary

NeuroSense is a multimodal machine learning project that trains separate models for EEG, MEG, MRI, speech, and facial input. Each modality follows a pipeline of data loading, preprocessing, feature extraction, training, evaluation, and artifact saving. EEG and MEG use structured numeric features with SVM, MRI and face use image preprocessing plus PCA and SVM, speech uses handcrafted audio features with speaker-holdout evaluation, and the final fusion stage combines probability outputs from multiple modalities using Logistic Regression and simple baselines. The backend loads the saved artifacts and applies the same preprocessing at inference time, making the project a complete end-to-end ML system.

## 19. Very Short Presentation Version

This project is a multimodal ML system that studies human-related signals using EEG, MEG, MRI, speech, and face data. Each modality is processed by its own pipeline and trained as a separate model. Their outputs are then combined using a fusion layer. The project demonstrates the full machine learning workflow from preprocessing and training to evaluation, artifact persistence, backend inference, and frontend interaction.
