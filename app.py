import streamlit as st
import numpy as np
import pandas as pd
import datetime
import requests
import os
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
# import cv2
from io import BytesIO


'''
# DEEPFAKE HUNTER APPLICATION
'''

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_files = st.file_uploader("Choose photos to upload", accept_multiple_files=False, type=['png', 'jpeg', 'jpg'])
st.set_option('deprecation.showfileUploaderEncoding', False) # Enabling the automatic file decoder

if st.button('AFFICHER l \'IMAGE'):

    image = Image.open(uploaded_files)

    fig = plt.figure(2, figsize=(200, 200))
    fig, ax = plt.subplots()


    st.image(image, width=500)



url = 'http://127.0.0.1:8000/predict'
if url == 'http://127.0.0.1:8000/predict':

    st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')

if st.button('predict'):

    # headers = {'Content-Type': 'image/jpeg'}
    response = requests.get(url,params={"image":uploaded_files}) # headers=headers)
    wb = response.json()

    # '''4. Let's retrieve the prediction from the **JSON** returned by the API...

    # ## Finally, we can display the prediction to the user
    # '''
    st.write('the prediction  is ', wb['result'])
else:
        st.write('I was not clicked 😞')
