from fastapi import FastAPI, File, UploadFile
from PIL import Image
import sys
import io
import numpy as np
import tensorflow as tf
from tensorflow import expand_dims

model = tf.keras.models.load_model('./model')
input_shape = model.layers[0].input_shape

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': "bienvenue dans l'index"}

@app.post('/predict')
async def predict_image(image: UploadFile = File(...)):

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        pic = np.array(pil_image)
        image_final = expand_dims(pic, 0)
        res = model.predict(image_final)
        print(res)
        return {"result": round(float(res[0][0]),2)}
    except:
        return {"error": str(sys.exc_info()[1])}
