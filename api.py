from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

import requests
# from model import load_data, initialize_model, fit_model, predict

from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.pyplot import imread
from tensorflow.image import resize
from tensorflow import expand_dims

model = models.load_model('./model-abou')

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': "bienvenue dans l'index"}

# url = 'http://localhost:8000/predict'

# params = "hello"

# response = requests.get(url, params=params).json()
# # response.json() #=> {wait: 64}

@app.get('/predict')
def predict_api(image):
    test_img = imread(image)
    test_img_resized = resize(test_img, [224,224])
    test_img_resized_scaled = test_img_resized/255.
    image_final = expand_dims(test_img_resized_scaled, 0)
    res = model.predict(image_final)
    return {"result": float(res[0][0])}
    # return {"result": str(image.size)}
