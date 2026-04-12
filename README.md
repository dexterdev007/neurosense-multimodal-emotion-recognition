# NeuroSense: Human Emotion Recognition System

NeuroSense is a beginner-friendly multimodal machine learning project. It trains separate models on five types of input:

- EEG
- MEG
- MRI
- Speech
- Face

After training the five base modules, the project also trains one fusion model that combines their outputs.

The ML side is intentionally kept classical and simple:

- `SVM` is used in the five base modules
- `LogisticRegression` is used in the fusion module
- preprocessing is kept explicit so it is easier to explain in a presentation or viva

## Beginner Learning Files

If you want the easiest explanation-first entry points, start with:

- `aboutproject/BEGINNER_PROJECT_GUIDE.md`
- `aboutproject/COPY_READY_PROJECT_EXPLANATION.md`
- `NeuroSense/notebooks/beginner_ml_pipeline_reference.py`
- `NeuroSense/notebooks/notebook_support.py`

## Folder Layout

```text
human emotion recognition system/
├── README.md
├── NeuroSense/
│   ├── artifacts/                 # saved .pkl model files
│   ├── datasets/                  # datasets used for training
│   ├── notebooks/                 # notebook 01 to 06
│   └── webdev/
│       ├── backend/               # FastAPI backend
│       └── frontend/              # HTML/CSS/JS frontend
└── render.yaml
```

## Quick Start

```bash
cd "human emotion recognition system"
source venv/bin/activate
pip install -r NeuroSense/webdev/backend/requirements.txt
```

Open Jupyter from the notebooks folder:

```bash
cd NeuroSense/notebooks
jupyter notebook
```

Run the notebooks in this order:

1. `01_EEG_Emotion_Recognition.ipynb`
2. `02_MEG_Emotion_Recognition.ipynb`
3. `03_MRI_Brain_Tumor.ipynb`
4. Speech (RAVDESS): `python NeuroSense/notebooks/train_speech_ravdess.py --label-mode sentiment --save`
5. `05_Face_Emotion_Recognition.ipynb`
6. `06_Fusion_Pipeline.ipynb`

Note:
`04_Speech_Emotion_Recognition.ipynb` is a legacy notebook built for the TESS dataset. The current project speech dataset is RAVDESS, so use `train_speech_ravdess.py` to regenerate speech artifacts.

Then start the backend:

```bash
cd NeuroSense/webdev/backend
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open the app at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Notebook Summary

| Notebook | Input Type | Main Model | Extra Step | Saved Files |
|---|---|---|---|---|
| 01 EEG | EEG CSV features | SVM | StandardScaler | model, scaler, label encoder |
| 02 MEG | MEG CSV features | SVM | StandardScaler | model, scaler, label encoder |
| 03 MRI | MRI images | SVM | StandardScaler + PCA | model, scaler, PCA, label encoder |
| 04 Speech | audio files (RAVDESS) | SVM | actor-holdout evaluation + StandardScaler | model, scaler, label encoder |
| 05 Face | face images | SVM | StandardScaler + PCA | model, scaler, PCA, label encoder |
| 06 Fusion | probabilities from base models | LogisticRegression | probability stacking | fusion model, label encoder |

## Common Code Used In The First Five Modules

The first five notebooks follow almost the same ML flow. Only the data source changes.

This is the main training pattern:

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

What each common part means:

- `X` means input features. This is what the model learns from.
- `y` means labels. This is the correct output for each row or file.
- `LabelEncoder()` changes text labels like `"POSITIVE"` or `"sad"` into numbers because models work with numeric targets.
- `train_test_split()` keeps some data aside for testing, so you can check the model on unseen samples.
- `StandardScaler()` brings numeric features to a similar scale. This helps SVM work better.
- `model.fit()` trains the model.
- `model.predict()` makes predictions on the test data.

All five modules also save files in the same way:

```python
joblib.dump(model, artifact_dir / "..._model.pkl")
joblib.dump(scaler, artifact_dir / "..._scaler.pkl")
joblib.dump(encoder, artifact_dir / "..._label_encoder.pkl")
```

Why these saved files matter:

- the model file stores the trained classifier
- the scaler file stores the exact scaling used during training
- the label encoder stores the mapping between numbers and class names

Without saving these objects, the backend would not be able to repeat the same prediction steps later.

## Common Helper Functions

Some helper functions are reused to keep the notebooks shorter.

- `bootstrap_notebook()` sets the main project paths.
- `extract_feature_dataset()` loops through files and builds the final `X` and `y`.
- `preprocess_mri_image()`, `preprocess_speech()`, and `preprocess_face_image()` convert raw files into numeric vectors.
- `map_emotion_to_sentiment()` is used when the original dataset labels need to be grouped into `NEGATIVE`, `NEUTRAL`, and `POSITIVE`.

Important idea:

The backend uses the same preprocessing functions as the notebooks. That keeps training-time preprocessing and prediction-time preprocessing consistent.

## What Changes In Each Module

### EEG

EEG already comes as tabular numeric data, so the notebook mainly:

- reads the CSV
- separates feature columns and the label column
- scales features
- trains SVM

### MEG

MEG is also tabular numeric data. If a CSV file is not present, the notebook creates a small synthetic dataset first so the training pipeline can still be demonstrated.

### MRI

MRI images cannot go directly into SVM. So the notebook first:

- converts each image to grayscale
- resizes it to `64 x 64`
- flattens the image into one long vector
- applies `PCA` to reduce the feature size
- trains SVM on the reduced features

Simple meaning of `PCA`:

- image vectors are very large
- PCA keeps the most useful information while reducing the number of columns
- smaller feature vectors make training easier

### Speech

Speech files are first converted into numeric audio features by `preprocess_speech()`. After that, the rest of the flow is the same as EEG and MEG:

- split
- scale
- train
- test
- save

The current speech dataset is **RAVDESS**, so evaluation is done with an **actor-holdout** split to avoid speaker leakage.

### Face

Face images follow a similar idea to MRI:

- convert image to grayscale
- resize to `48 x 48`
- flatten pixels
- add a few simple statistical values
- reduce with `PCA`
- train SVM

This module also maps many original emotions into three final sentiment classes for a simpler final output.

## Fusion Module

The fusion notebook is the last step. It does not train from raw files directly.

Instead, it:

- loads the already trained base models
- gets probability outputs from each model
- converts those outputs into the same 3-class sentiment format
- joins those probabilities into one new feature vector
- trains one `LogisticRegression` model on top

So the fusion model is basically a model that learns from the predictions of other models.

## Backend Connection

The backend loads the saved `.pkl` files from `NeuroSense/artifacts/`.

That is why the notebook filenames and saved artifact names should stay consistent:

- `eeg_model.pkl`
- `eeg_scaler.pkl`
- `eeg_label_encoder.pkl`
- and the matching files for the other modules

If you retrain a notebook, restart the backend so it reloads the new artifacts.

## Why This Project Uses Basic ML

This project uses classical ML on purpose:

- easier to understand than a full deep learning pipeline
- easier to explain line by line
- faster to train on a normal laptop
- enough to demonstrate preprocessing, model training, evaluation, and deployment

If you want to explain the project simply, the shortest summary is:

1. each module turns its input into numeric features
2. an SVM learns patterns from those features
3. the trained objects are saved
4. the backend loads the saved objects and predicts on new input
5. the fusion model combines outputs from the separate modules
