# NeuroSense Beginner Project Guide

This file explains the full NeuroSense project in simple language:

- what the project does
- how the ML pipeline works
- why each model and preprocessing step is used
- how the code is organized
- how the backend uses the trained artifacts

It is written for beginners, project reports, and viva preparation.

## 1. Project Idea

NeuroSense is a **multimodal machine learning system**.

That means it does not rely on only one type of input. Instead, it uses multiple sources:

- EEG
- MEG
- MRI
- Speech
- Face

Each modality gets its own model first. After that, the project combines multiple model outputs in a **fusion stage** to make one final sentiment-style prediction.

The main idea is:

1. read raw data
2. convert raw data into numeric features
3. train a classifier
4. evaluate performance
5. save artifacts
6. use the saved artifacts in the backend for prediction

## 2. Main Goal Of The Project

The project demonstrates how a complete ML system is built from start to end:

- training notebooks
- saved model artifacts
- backend prediction API
- frontend interface

It is not only about accuracy.

It is also about understanding the full pipeline:

- data collection
- preprocessing
- feature extraction
- model training
- validation
- artifact saving
- deployment-ready inference

## 3. High-Level Working

The full working of the project is:

1. A training notebook loads one dataset.
2. The notebook preprocesses the data into features.
3. The notebook trains a model.
4. The notebook evaluates the model on unseen data.
5. The notebook saves the trained model and related helper objects.
6. The backend loads those saved files.
7. When a user uploads new data, the backend repeats the same preprocessing.
8. The backend sends the processed features into the saved model.
9. The backend returns the prediction, confidence, and class probabilities.

So the project has two connected parts:

- **training time**
- **prediction time**

The most important rule is:

> preprocessing at prediction time must match preprocessing at training time

That is why files like `preprocessors.py`, scalers, PCA objects, and label encoders are so important.

## 4. Project Folder Meaning

Important folders:

- `NeuroSense/notebooks/`
  Contains the six main training notebooks.

- `NeuroSense/artifacts/`
  Contains the saved models and helper files such as scalers, PCA objects, label encoders, and metadata.

- `NeuroSense/webdev/backend/`
  Contains the FastAPI backend that loads the trained artifacts and serves predictions.

- `NeuroSense/webdev/frontend/`
  Contains the simple frontend interface.

## 5. Notebook Order

The notebooks are meant to be run in this order:

1. `01_EEG_Emotion_Recognition.ipynb`
2. `02_MEG_Emotion_Recognition.ipynb`
3. `03_MRI_Brain_Tumor.ipynb`
4. Speech (RAVDESS): `python NeuroSense/notebooks/train_speech_ravdess.py --label-mode sentiment --save`
5. `05_Face_Emotion_Recognition.ipynb`
6. `06_Fusion_Pipeline.ipynb`

Note:
`04_Speech_Emotion_Recognition.ipynb` is a legacy notebook built for the TESS dataset. The current project speech dataset is RAVDESS, so use `train_speech_ravdess.py` to regenerate speech artifacts.

Why this order matters:

- notebooks 1 to 5 train the base models
- notebook 6 uses outputs from those trained base models

So fusion depends on the earlier notebooks.

## 6. ML Theory Used In This Project

### 6.1 Supervised Learning

This project uses **supervised learning**.

That means:

- the input data is already labeled
- the model learns the mapping from input to output

Examples:

- EEG features -> `NEGATIVE`, `NEUTRAL`, or `POSITIVE`
- face image -> mapped sentiment label
- speech audio -> emotional class

### 6.2 Classification

Most tasks in this project are **classification tasks**.

Classification means:

- the output is a category
- not a continuous number

Examples of classes:

- `NEGATIVE`
- `NEUTRAL`
- `POSITIVE`
- MRI tumor classes like `glioma`, `meningioma`, `pituitary`, `no_tumor`

### 6.3 Features And Labels

Two fundamental ML terms:

- `X` = input features
- `y` = target labels

Examples:

- EEG CSV columns are features
- the `label` column is the target
- MRI pixel values become features after flattening
- speech audio descriptors become features after extraction

### 6.4 Train-Test Split

The model should not be evaluated on the exact same data it learned from.

That is why the pipeline uses:

- training set
- test set

Training set:
- used to learn patterns

Test set:
- used to check generalization on unseen samples

### 6.5 StandardScaler

`StandardScaler` is used in multiple notebooks.

Why:

- some features may be very large
- some may be very small
- SVM works better when features are on a similar scale

What it does:

- subtracts the mean
- divides by the standard deviation

Important:

- fit the scaler on training data
- use the same scaler on test data and future user inputs

### 6.6 Label Encoding

ML models usually expect numeric target values, not strings.

So labels like:

- `"happy"`
- `"sad"`
- `"POSITIVE"`

are converted into integers by `LabelEncoder`.

The saved label encoder is later used to convert numeric model outputs back into human-readable labels.

### 6.7 SVM

Most base models in this project use **SVM** with an RBF kernel.

Why SVM:

- good for medium-sized classical ML problems
- works well on numeric feature vectors
- easier to explain than deep neural networks
- suitable for a beginner-friendly academic project

What the RBF kernel does:

- helps the model learn non-linear decision boundaries
- useful when classes are not separable by a straight line

### 6.8 PCA

`PCA` is used in image-heavy modalities like MRI and Face.

Why PCA is needed:

- images have many pixel features
- too many features can increase training difficulty
- some features are redundant

What PCA does:

- compresses the feature space
- keeps the directions with the most variance
- reduces dimensionality while preserving important structure

In simple words:

PCA creates a smaller feature representation that still contains much of the useful information.

### 6.9 Logistic Regression In Fusion

The fusion stage uses `LogisticRegression`.

Why:

- the input to fusion is already a compact probability vector
- logistic regression is simple, fast, and interpretable
- it works well as a meta-model for combining predictions

### 6.10 Overfitting

A model can memorize the training data too much.

That is called **overfitting**.

Common sign:

- very high train accuracy
- much lower test accuracy

That is why the notebooks print train-test gaps and evaluation results.

## 7. Common ML Pipeline Used In The Project

Most notebooks follow this pattern:

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

model = SVC(kernel="rbf", C=5.0, probability=True, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

Meaning of each step:

1. `LabelEncoder()`
   changes text labels into integer labels

2. `train_test_split()`
   keeps some data aside for fair testing

3. `StandardScaler()`
   scales the numeric feature values

4. `SVC(...)`
   creates the SVM classifier

5. `model.fit(...)`
   trains the model

6. `model.predict(...)`
   predicts classes for unseen test data

## 8. Modality-Wise Technical Explanation

### 8.1 EEG Module

Notebook:
- `01_EEG_Emotion_Recognition.ipynb`

Input:
- EEG CSV file

Why it is simpler:
- EEG features are already numeric
- there is no image decoding or audio processing

Pipeline:

1. load CSV with pandas
2. separate features and label column
3. fill missing numeric values
4. encode labels
5. split train and test data
6. scale features
7. train SVM
8. evaluate accuracy and confusion matrix
9. save model, scaler, and label encoder

Saved files:

- `eeg_model.pkl`
- `eeg_scaler.pkl`
- `eeg_label_encoder.pkl`

### 8.2 MEG Module

Notebook:
- `02_MEG_Emotion_Recognition.ipynb`

Input:
- MEG feature CSV

Special point:
- if a real MEG CSV is missing, the notebook creates a synthetic dataset so the pipeline can still be demonstrated

Why that matters:
- the code path can still be shown in class or viva
- but synthetic accuracy should not be presented as real-world performance

Pipeline:

1. load real or synthetic features
2. encode labels
3. split dataset
4. scale features
5. train SVM
6. evaluate test accuracy
7. save artifacts and metadata

### 8.3 MRI Module

Notebook:
- `03_MRI_Brain_Tumor.ipynb`

Input:
- MRI images

Problem:
- SVM cannot directly work on raw image files

So the notebook converts each image into a numeric feature vector.

Pipeline:

1. load MRI image paths
2. decode image
3. convert to grayscale
4. resize to `64 x 64`
5. flatten the pixels into one long vector
6. scale features
7. apply PCA
8. train SVM on PCA output
9. evaluate on test images
10. save model, scaler, PCA, and label encoder

Why grayscale:

- simplifies the image
- reduces unnecessary color channels

Why flattening:

- classical ML models need tabular numeric vectors

Why PCA:

- a `64 x 64` image gives 4096 raw pixel features
- PCA compresses this into a smaller representation

### 8.4 Speech Module

Training script:
- `NeuroSense/notebooks/train_speech_ravdess.py`

Input:
- speech audio files

Important concept:

Audio files are not used directly.
They are converted into **hand-crafted audio features**.

The shared preprocessing function extracts 162 values, including:

- MFCC means and standard deviations
- chroma features
- mel-spectrogram summaries
- spectral contrast
- tonnetz
- zero crossing rate
- RMS energy
- spectral rolloff

Why these features are used:

- classical models like SVM need fixed-length numeric vectors
- these features summarize the frequency and energy behavior of speech

Very important evaluation detail:

The speech module uses a **speaker-aware evaluation** on the RAVDESS dataset.

Why this is better:

- if the same actor appears in both training and testing, the model may learn actor identity instead of emotion
- using an **actor-holdout** split gives a more honest estimate of generalization

The training script also reports a GroupKFold score (speaker-aware cross-validation) for a more stable estimate.

This is a strong technical point in the project because it shows awareness of **data leakage** while keeping the code simple.

### 8.5 Face Module

Notebook:
- `05_Face_Emotion_Recognition.ipynb`

Input:
- face images

Pipeline:

1. load train and test image paths
2. map detailed emotions into broader sentiment classes
3. decode image
4. convert to grayscale
5. resize to `48 x 48`
6. flatten pixel values
7. normalize pixels
8. add simple image statistics
9. scale features
10. apply PCA
11. train SVM
12. evaluate and save artifacts

Extra points:

- PCA reduces dimensionality

Why this is still classical ML:

- no CNN is used
- instead, image pixels plus simple statistics are used as features

This makes the system easier to explain, though not as strong as a deep learning vision model.

### 8.6 Fusion Module

Notebook:
- `06_Fusion_Pipeline.ipynb`

This notebook is different from the others.

It does not learn from raw EEG, raw audio, or raw images directly.

Instead, it learns from the **probability outputs** of already trained base models.

This is called **late fusion** or **stacking-style fusion**.

Pipeline:

1. load base models and their helper artifacts
2. run each base model on its own evaluation data
3. collect class probability outputs
4. convert outputs into the same 3-sentiment space
5. stack the modality probabilities together
6. train a logistic regression meta-model
7. compare it with simple averaging and majority voting
8. save fusion model and metadata

Why fusion is useful:

- one modality may be noisy
- another modality may still be confident
- combining multiple signals can improve robustness

## 9. Why Saved Artifacts Matter

When a notebook finishes training, it saves several files.

These usually include:

- model
- scaler
- label encoder
- PCA object for image pipelines
- metadata

Why each one matters:

- **model**: learned decision rules
- **scaler**: exact feature scaling learned from training data
- **label encoder**: mapping between integers and class names
- **PCA**: exact dimensionality reduction mapping
- **metadata**: notes about evaluation, source, and honesty markers

Without these files, the backend could not reproduce the training-time behavior.

## 10. Backend Working

Important backend files:

- `webdev/backend/app.py`
- `webdev/backend/utils/model_loader.py`
- `webdev/backend/utils/preprocessors.py`
- `webdev/backend/utils/metadata_loader.py`
- router files in `webdev/backend/routers/`

Backend flow:

1. startup loads all saved artifacts
2. user sends input to an API route
3. route calls the correct preprocessing function
4. processed features are passed through scaler and optional PCA
5. model predicts class and probabilities
6. backend returns JSON response

Example:

- face image upload
- backend runs `preprocess_face_image()`
- scaler transforms the features
- PCA reduces dimensions
- face SVM predicts the class
- backend returns prediction and confidence

## 11. Code-Level Explanation Of Important Shared Files

### `notebook_support.py`

This file keeps repeated notebook helper logic in one place.

It handles:

- project path setup
- cache directory creation
- image/audio file collection
- duplicate removal
- converting file lists into feature matrices

This keeps notebooks shorter and easier to read.

### `preprocessors.py`

This file is one of the most important technical files in the whole project.

It defines how raw user input is converted into model-ready features.

Functions include:

- `preprocess_eeg()`
- `preprocess_meg()`
- `preprocess_mri_image()`
- `preprocess_speech()`
- `preprocess_face_image()`

If these functions do not match the notebook logic, predictions can become incorrect.

### `model_loader.py`

This file loads all saved artifacts from disk into memory.

Why this is useful:

- models do not need to be reloaded for every request
- the API becomes faster after startup

### `emotion_utils.py`

This file helps map many detailed labels into common sentiment groups:

- `NEGATIVE`
- `NEUTRAL`
- `POSITIVE`

It is especially useful for the fusion stage because different datasets may use different emotion names.

## 12. Important Technical Concepts For Viva

These are strong concepts to explain if someone asks technical questions:

- multimodal machine learning
- supervised classification
- feature extraction
- dimensionality reduction with PCA
- train-test split
- cross-validation
- speaker leakage in speech emotion recognition
- overfitting
- probability-based late fusion
- artifact persistence with `joblib`
- consistency between training-time and inference-time preprocessing

## 13. Strengths Of This Project

- covers the full ML lifecycle
- supports multiple modalities
- uses explainable classical ML methods
- saves reusable inference artifacts
- has backend and frontend integration
- includes honest evaluation notes in metadata

## 14. Limitations Of This Project

- classical ML on images is weaker than CNN-based deep learning
- fusion is limited by the quality of base models
- MEG may rely on synthetic data if a real dataset is not present
- different datasets have different label styles and noise levels
- real-world multimodal synchronization is not fully modeled here

## 15. How To Explain The Whole Project In One Paragraph

NeuroSense is a multimodal machine learning project that trains separate classical ML models for EEG, MEG, MRI, speech, and face data, then combines multiple model outputs through a fusion stage. Each notebook follows a standard pipeline of data loading, preprocessing, feature extraction, scaling, model training, evaluation, and artifact saving. The backend later loads those saved artifacts and applies the same preprocessing steps to new user inputs, which makes the project a full end-to-end ML system rather than only a set of experiments.

## 16. Best Short Explanation Of The ML Pipeline

The ML pipeline in this project is:

1. collect raw data
2. convert raw data into numeric features
3. encode labels
4. split into training and testing data
5. scale features
6. reduce dimensions when needed
7. train a classifier
8. evaluate on unseen data
9. save artifacts
10. reuse those artifacts during backend inference

## 17. Suggested Files To Read First

If someone is new to this project, the best order is:

1. `README.md`
2. `aboutproject/BEGINNER_PROJECT_GUIDE.md`
3. `NeuroSense/notebooks/beginner_ml_pipeline_reference.py`
4. `NeuroSense/notebooks/01_EEG_Emotion_Recognition.ipynb`
5. `NeuroSense/webdev/backend/utils/preprocessors.py`
6. `NeuroSense/webdev/backend/utils/model_loader.py`

That order gives both the concept and the code.
