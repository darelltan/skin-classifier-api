from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io, base64, cv2
import os
from gradcam import get_prediction, make_gradcam_heatmap, overlay_heatmap

app = FastAPI(title="Skin Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

if os.environ.get("TESTING") != "1":
    print("Loading model...")
    model = tf.keras.models.load_model("models/final_model.keras")
    print("Model loaded.")
else:
    model = None
    print("Skipping model load in test mode.")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

@app.get("/")
def root():
    return {"status": "ok", "message": "Skin classifier API running"}

@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Only JPEG, PNG, and WebP images accepted")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large — max 10MB")

    try:
        img   = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        arr   = np.array(img)
        batch = np.expand_dims(arr, axis=0).astype("float32")

        result  = get_prediction(batch, model)
        heatmap = make_gradcam_heatmap(batch, model)
        overlay = overlay_heatmap(arr, heatmap)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buf      = cv2.imencode('.jpg', overlay_bgr)
        heatmap_b64 = base64.b64encode(buf).decode()

        return {**result, "heatmap_base64": heatmap_b64}

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")