from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "Sleep & Awake Classification API is running"}


# async def predict() → function that handles the request
# File(...): tells FastAPI that the input comes from a file upload

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()  # "Take the uploaded file and read its contents."
    image_array = np.frombuffer(image_bytes, np.uint8)  # Now we convert those raw bytes into a NumPy array.
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)   # takes encoded image data (bytes/NumPy array) and converts it into an OpenCV image

    results = model(img)
    for result in results:

        cls = result.probs.top1
        conf = result.probs.top1conf
        name = model.names[cls]

        return {"prediction": name, "confidence": float(conf)}