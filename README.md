# Skin Condition Classifier — Backend API

A FastAPI backend serving a CNN-based skin lesion classifier with Grad-CAM interpretability visualisation. Built as a portfolio project for SIT ICT (Software Engineering) application.

![CI Status](https://github.com/darelltan/skin-classifier-api/actions/workflows/test.yml/badge.svg)

## Live Demo
- **Full app:** https://skin-classifier-ui.vercel.app
- **Gradio demo:** https://huggingface.co/spaces/renomaaaa/skin-classifier
- **API docs:** https://renomaaaa-skin-classifier-api.hf.space/docs

## Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Basal Cell Carcinoma | 0.70 | 0.80 | 0.75 |
| Benign Keratosis | 0.53 | 0.62 | 0.57 |
| Melanoma | 0.59 | 0.50 | 0.54 |
| Melanocytic Nevus | 0.70 | 0.59 | 0.64 |
| **Overall** | **0.63** | **0.63** | **0.62** |

Test accuracy: **62.7%** (vs 25% random baseline on 4-class problem)

## Confusion matrix
![Confusion matrix](confusion_matrix.png)

## Grad-CAM visualisations
![Grad-CAM examples](gradcam_examples.png)

## Technical decisions

**Transfer learning with EfficientNetB0** — The ISIC dataset provides ~1,500 images per class after balancing, which is insufficient to train a deep CNN from scratch without severe overfitting. EfficientNetB0 pretrained on ImageNet provides strong feature extraction foundations that are fine-tuned for dermoscopic features in Phase 2 training.

**Two-phase training** — Phase 1 trains only the classification head with the base model frozen (learning rate 1e-3). Phase 2 unfreezes the last 30 layers of EfficientNetB0 and fine-tunes at a very low learning rate (1e-5) to adapt pretrained features without destroying them.

**Class balancing** — The raw ISIC 2019 dataset is heavily imbalanced (nevus: 12,875 images vs basal cell: 3,323). Each class was capped at 1,500 images to prevent the model defaulting to majority-class predictions.

**Grad-CAM interpretability** — In medical AI, interpretability is essential. Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which regions of the input image most influenced the prediction. For melanoma, the model correctly focuses on irregular borders and pigmentation variation — the same features used in clinical diagnosis.

**Melanoma recall (0.50)** — The weakest metric and the most clinically important. Missing a melanoma (false negative) is more dangerous than a false alarm. This reflects a known challenge in dermoscopy — melanoma and keratosis share visual features that confuse models trained on limited data.

## Dataset

ISIC 2019 Challenge dataset (publicly available). 4 classes, 1,500 images per class after balancing. 70/15/15 train/val/test split.

| Class | ISIC label | Raw count | After balancing |
|-------|-----------|-----------|-----------------|
| Melanoma | MEL | 4,522 | 1,500 |
| Melanocytic Nevus | NV | 12,875 | 1,500 |
| Basal Cell Carcinoma | BCC | 3,323 | 1,500 |
| Benign Keratosis | BKL | 2,624 | 1,500 |

## Tech stack
- Python 3.11
- TensorFlow / Keras (EfficientNetB0)
- FastAPI
- OpenCV (Grad-CAM overlay)
- Docker (Hugging Face deployment)
- pytest + GitHub Actions (CI pipeline)

## Testing & CI/CD

Automated test suite with pytest covering all main API endpoints. GitHub Actions runs the full test suite on every push to main.

### Test coverage
- `GET /` — root endpoint health check
- `POST /predict` — missing file returns 422
- `POST /predict` — wrong file type returns 400
- `POST /predict` — valid JPEG returns prediction, confidence scores, and Grad-CAM heatmap

### Running tests locally
```bash
pip install pytest httpx
pytest test_main.py -v
```

Tests use mocking to skip model loading — no GPU or model file needed to run the test suite.

## API

### POST /predict

Accepts a dermoscopy image, returns prediction and Grad-CAM heatmap.

**Request:** multipart/form-data with `file` field (JPEG/PNG/WebP, max 10MB)

**Response:**
```json
{
  "class": "melanoma",
  "confidence": 94.2,
  "all_scores": {
    "basal_cell": 1.1,
    "keratosis": 2.3,
    "melanoma": 94.2,
    "nevus": 2.4
  },
  "heatmap_base64": "/9j/4AAQ..."
}
```

## Run locally

```bash
git clone https://github.com/darelltan/skin-classifier-api
cd skin-classifier-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Place `final_model.keras` in the `models/` folder before running.

## Disclaimer

This is a research prototype built for a student portfolio. It is not a medical device and must not be used for clinical diagnosis. Always consult a qualified dermatologist.
