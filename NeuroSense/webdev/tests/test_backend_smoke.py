from __future__ import annotations

import math
import unittest
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "webdev" / "backend"

import sys

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import app  # noqa: E402
from utils.metadata_loader import load_modality_metadata  # noqa: E402
from utils.model_loader import get, load_all_models  # noqa: E402


DATASETS_DIR = PROJECT_ROOT / "datasets"


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _assert_probability_map(testcase: unittest.TestCase, payload: dict) -> None:
    testcase.assertIn("prediction", payload)
    testcase.assertIn("confidence", payload)
    testcase.assertIn("probabilities", payload)
    testcase.assertIsInstance(payload["probabilities"], dict)
    if payload["probabilities"]:
        total = sum(float(value) for value in payload["probabilities"].values())
        testcase.assertTrue(math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2))


class BackendSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_all_models()

        eeg_df = pd.read_csv(DATASETS_DIR / "eeg" / "eeg" / "emotions.csv")
        cls.eeg_features = eeg_df.drop(columns=["label"]).iloc[0].astype(float).tolist()

        cls.meg_features = [0.0] * int(get("meg", "scaler").n_features_in_)

        cls.mri_path = next((DATASETS_DIR / "mri" / "mri" / "Training").rglob("*.jpg"))
        cls.face_path = next((DATASETS_DIR / "face" / "archive" / "test").rglob("*.jpg"))
        cls.speech_path = next((DATASETS_DIR / "speech").rglob("*.wav"))

    def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        async def _send() -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)

        import asyncio

        return asyncio.run(_send())

    def test_health(self) -> None:
        response = self._request("GET", "/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_eeg_predict(self) -> None:
        response = self._request("POST", "/api/eeg/predict", json={"features": self.eeg_features})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["feature_count"], len(self.eeg_features))
        _assert_probability_map(self, payload)

    def test_meg_predict(self) -> None:
        response = self._request("POST", "/api/meg/predict", json={"features": self.meg_features})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["feature_count"], len(self.meg_features))
        self.assertIn("data_source", payload)
        self.assertIn("data_source_note", payload)
        _assert_probability_map(self, payload)

    def test_mri_predict(self) -> None:
        response = self._request(
            "POST",
            "/api/mri/predict",
            files={"file": (self.mri_path.name, _read_bytes(self.mri_path), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        _assert_probability_map(self, response.json())

    def test_face_predict(self) -> None:
        response = self._request(
            "POST",
            "/api/face/predict",
            files={"file": (self.face_path.name, _read_bytes(self.face_path), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        _assert_probability_map(self, response.json())

    def test_speech_predict(self) -> None:
        response = self._request(
            "POST",
            "/api/speech/predict",
            files={"file": (self.speech_path.name, _read_bytes(self.speech_path), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("holdout", str(payload.get("evaluation_method", "")).lower())
        self.assertTrue(payload.get("evaluation_note"))
        _assert_probability_map(self, payload)

    def test_fusion_accepts_single_face_modality(self) -> None:
        response = self._request(
            "POST",
            "/api/fusion/predict",
            json={
                "modalities": {
                    "face": {
                        "prediction": "POSITIVE",
                        "confidence": 0.81,
                        "probabilities": {"POSITIVE": 0.81, "NEUTRAL": 0.12, "NEGATIVE": 0.07},
                    }
                }
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["active_modalities"], ["face"])
        self.assertIn("Average sentiment aggregation", payload["model_used"])
        _assert_probability_map(self, payload)

    def test_fusion_accepts_partial_subset(self) -> None:
        response = self._request(
            "POST",
            "/api/fusion/predict",
            json={
                "modalities": {
                    "eeg": {"prediction": "POSITIVE", "confidence": 0.7},
                    "mri": {"prediction": "pituitary", "confidence": 0.9},
                    "face": {"prediction": "NEGATIVE", "confidence": 0.8},
                }
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertCountEqual(payload["active_modalities"], ["eeg", "mri", "face"])
        self.assertIn("Average sentiment aggregation", payload["model_used"])
        _assert_probability_map(self, payload)

    def test_fusion_accepts_full_core_modalities(self) -> None:
        response = self._request(
            "POST",
            "/api/fusion/predict",
            json={
                "modalities": {
                    "eeg": {"prediction": "POSITIVE", "confidence": 0.7},
                    "meg": {"prediction": "NEGATIVE", "confidence": 0.65},
                    "speech": {"prediction": "NEUTRAL", "confidence": 0.8},
                    "face": {"prediction": "POSITIVE", "confidence": 0.75},
                }
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertCountEqual(payload["active_modalities"], ["eeg", "meg", "speech", "face"])
        fusion_metadata = load_modality_metadata("fusion")
        if fusion_metadata.get("meta_model_adds_value", True):
            self.assertEqual(payload["model_used"], "LogisticRegression (meta)")
            self.assertEqual(payload["fusion_method"], "trained_meta_model")
        else:
            self.assertIn("Average sentiment aggregation", payload["model_used"])
            self.assertEqual(payload["fusion_method"], "average_aggregation")
            self.assertTrue(payload.get("meta_model_disabled_reason"))
        _assert_probability_map(self, payload)


if __name__ == "__main__":
    unittest.main()
