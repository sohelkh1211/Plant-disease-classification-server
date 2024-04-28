from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

class_names = ['Apple Black rot', 'Apple Healthy', 'Apple Scab', 'Cedar apple rust', 'Cherry Healthy', 'Cherry Powdery mildew', 'Corn Common rust', 'Corn Gray leaf spot', 'Corn Healthy', 'Corn Northern Leaf Blight', 'Grape Black Measles', 'Grape Black rot', 'Grape Healthy', 'Grape Isariopsis Leaf Spot', 'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'TomatoLate blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Twos potted spider mite', 'Tomato Target Spot', 'Tomato Tomato YellowLeaf Curl Virus', 'Tomato Tomato mosaic virus', 'Tomato healthy']
model = tf.keras.models.load_model("models/my_model.h5")
print(tf.__version__)
app = FastAPI()

def build_cors_preflight_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    }

origins = [
    "https://plant-disease-classification.netlify.app",
    "http://localhost",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)).convert("RGB").resize((256, 256)))
    return image

@app.get("/ping")
async def ping():
    return "Hello, I'm alive"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,axis=0)
    prediction = model.predict(image_batch)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(np.max(prediction[0])*100,2)
    response = {
        "class": predicted_class,
        "confidence": float(confidence)
    }
    return JSONResponse(
        content=response,
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.options("/predict")
def options_predict(request: Request):
    return JSONResponse(content={}, headers=build_cors_preflight_headers())
