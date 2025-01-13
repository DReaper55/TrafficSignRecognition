import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile

from src.evaluation.test_model import load_and_predict, gtsrb_classes
from src.utils.helpers import normalize_images

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile):
    # Read image
    content = await file.read()
    np_image = np.frombuffer(content, np.uint8)


    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Preprocess image
    img = cv2.resize(img, (64, 64))  # Resize to model's expected input
    img = normalize_images(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    predictions = load_and_predict(img)
    predicted_class = gtsrb_classes[predictions[0]]

    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
