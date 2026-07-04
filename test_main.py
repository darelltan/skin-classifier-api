import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Mock the model loading so tests don't need the actual keras file
with patch("main.tf.keras.models.load_model") as mock_load:
    mock_load.return_value = MagicMock()
    from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data

def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422   # missing required field

def test_predict_wrong_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400

def test_predict_valid_image():
    # Create a tiny valid JPEG in memory
    from PIL import Image
    import io
    img = Image.new("RGB", (100, 100), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    with patch("main.get_prediction") as mock_pred, \
         patch("main.make_gradcam_heatmap") as mock_cam, \
         patch("main.overlay_heatmap") as mock_overlay:

        mock_pred.return_value = {
            "class": "melanoma",
            "confidence": 94.2,
            "all_scores": {
                "basal_cell": 1.1,
                "keratosis": 2.3,
                "melanoma": 94.2,
                "nevus": 2.4
            }
        }
        mock_cam.return_value = np.zeros((7, 7), dtype=np.float32)
        mock_overlay.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["class"] == "melanoma"
    assert data["confidence"] == 94.2
    assert "heatmap_base64" in data
    assert "all_scores" in data