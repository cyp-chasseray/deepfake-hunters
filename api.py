from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import sys
import io
import numpy as np

import requests
# from model import load_data, initialize_model, fit_model, predict

from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.pyplot import imread
from tensorflow.image import resize
from tensorflow import expand_dims

model = models.load_model('./model-abou')
input_shape = model.layers[0].input_shape

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': "bienvenue dans l'index"}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

@app.post('/predict')
async def predict_image(image: UploadFile = File(...)):

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        # .resize((input_shape[1], input_shape[2]))
        pic = np.array(pil_image)
        image_scaled = pic/255.
        image_final = expand_dims(image_scaled, 0)
        res = model.predict(image_final)
        print(res)
        return {"result": float(res[0][0])}

    except:
        e = sys.exc_info()[1]
		# raise HTTPException(status_code=500, detail=str(e))
