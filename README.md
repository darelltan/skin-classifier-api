# Skin Condition Classifier — Backend API

A FastAPI backend for classifying skin lesions using an EfficientNetB0 CNN with Grad-CAM visualisations for model interpretability.

Built to explore deep learning, medical image classification, and production deployment using FastAPI, Docker, and GitHub Actions.

![CI Status](https://github.com/darelltan/skin-classifier-api/actions/workflows/test.yml/badge.svg)

## Live Demo
- **Full app:** https://skin-classifier-ui.vercel.app
- **Gradio demo:** https://huggingface.co/spaces/renomaaaa/skin-classifier
- **API docs:** https://renomaaaa-skin-classifier-api.hf.space/docs

## Features
- Four-class skin lesion classification using EfficientNetB0
- Confidence scores for all predicted classes
- Grad-CAM heatmap generation
- REST API built with FastAPI
- Interactive Swagger documentation
- Docker deployment on Hugging Face Spaces
- Automated testing with pytest
- Continuous Integration using GitHub Actions

## Architecture

```text
                Client
                   │
                   ▼
           FastAPI Backend
                   │
                   ▼
      Image Preprocessing
                   │
                   ▼
      EfficientNetB0 Model
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Predicted Class      Confidence Scores
        │
        ▼
 Grad-CAM Heatmap
```

## Tech stack
- Python 3.11
- TensorFlow / Keras (EfficientNetB0)
- FastAPI
- OpenCV (Grad-CAM overlay)
- Docker (Hugging Face deployment)
- pytest + GitHub Actions (CI pipeline)

## Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Basal Cell Carcinoma | 0.70 | 0.80 | 0.75 |
| Benign Keratosis | 0.53 | 0.62 | 0.57 |
| Melanoma | 0.59 | 0.50 | 0.54 |
| Melanocytic Nevus | 0.70 | 0.59 | 0.64 |
| **Overall** | **0.63** | **0.63** | **0.62** |

Test accuracy: **62.7%** (vs **25%** random baseline on a four-class classification task).

Since this is a medical image classification problem, precision, recall, and F1-score provide a better measure of performance than overall accuracy alone.

## Confusion matrix
![Confusion matrix](confusion_matrix.png)

## Grad-CAM visualisations
![Grad-CAM examples](gradcam_examples.png)

## Technical decisions

**Transfer learning with EfficientNetB0** — The balanced ISIC dataset contains around 1,500 images per class, which is still relatively small for training a deep convolutional neural network from scratch. Doing so would likely lead to overfitting, where the model memorizes the training images instead of learning general patterns. To avoid this, I used EfficientNetB0 pretrained on ImageNet. Since it has already learned useful visual features such as edges, shapes, and textures, the model only needs to adapt these features to recognise different types of skin lesions.


**Two-phase training** — I trained the model in two stages.
Phase 1: I froze the pretrained EfficientNetB0 layers and trained only the final classification head using a learning rate of 1e-3. This allowed the new classifier to learn the skin lesion classes without changing the pretrained features.
Phase 2: After the classifier had stabilised, I unfroze the last 30 layers of EfficientNetB0 and fine-tuned them with a much smaller learning rate (1e-5). This helps the model adapt to dermoscopic images while reducing the risk of overwriting the useful features learned during pretraining.

**Class balancing** — The original ISIC 2019 dataset is highly imbalanced. For example, the nevus class contains over 12,000 images, while classes like basal cell carcinoma have only around 3,000. To reduce this imbalance, I limited every class to 1,500 images. Although this meant using fewer total images, it encouraged the model to learn each class more evenly instead of favouring the majority class.

**Grad-CAM interpretability** — Medical AI models should not behave like "black boxes." To better understand what the model is looking at, I used Grad-CAM (Gradient-weighted Class Activation Mapping). Grad-CAM generates a heatmap showing which parts of the image contributed most to the prediction. For melanoma cases, the model often focuses on irregular borders and uneven pigmentation, which are also important visual cues used by dermatologists.


**Melanoma recall (0.50)** — One of the biggest challenges in this project is detecting melanoma. The model achieved a recall of 0.50, meaning it successfully identified about half of the melanoma cases in the test set. This is the weakest class in the model, but it is also the most important clinically because missing a melanoma (a false negative) can have serious consequences. One reason for this lower performance is that melanoma often shares visual characteristics with benign lesions such as keratosis, making them difficult to distinguish, especially with a relatively small training dataset.

## Dataset
The model was trained using the publicly available ISIC 2019 Challenge dataset. To reduce class imbalance, each class was capped at 1,500 images before applying a 70/15/15 train/validation/test split.

| Class | ISIC label | Raw count | After balancing |
|-------|-----------|-----------|-----------------|
| Melanoma | MEL | 4,522 | 1,500 |
| Melanocytic Nevus | NV | 12,875 | 1,500 |
| Basal Cell Carcinoma | BCC | 3,323 | 1,500 |
| Benign Keratosis | BKL | 2,624 | 1,500 |

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

## Repository Structure
```text
skin-classifier-api/
├── main.py                 # FastAPI application
├── models/
│   └── final_model.keras
├── utils/                  # Image preprocessing and Grad-CAM utilities
├── tests/
│   └── test_main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## What I Learned
Through this project I gained experience with:

- Transfer learning using EfficientNetB0
- Handling class imbalance in medical datasets
- Interpreting CNN predictions with Grad-CAM
- Building REST APIs with FastAPI
- Docker deployment
- Writing automated API tests

## Future Improvements
- Improve melanoma recall through additional training data
- Experiment with larger EfficientNet variants
- Add lesion segmentation before classification
- Optimise inference speed for CPU deployment
- Support batch predictions

## Disclaimer
This is a research prototype built for a student portfolio. It is not a medical device and must not be used for clinical diagnosis. Always consult a qualified dermatologist.
