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

    files={'image' : uploaded_files}

    response = requests.post(url, files=files)
    if response.status_code == 200:
        wb = response.json()
        # wb = response.text
        # st.write(wb + '  ' + str(response.status_code)
        # st.write('the prediction  is ', wb['result'])
        st.write('the prediction  is ', wb["result"])
    else:
        st.write('the response.status_code  is ', str(response.status_code) + '   ' + response.text)
        st.write('the response.status_code  is ', str(response.status_code) + '   ' + uploaded_files.name)
else:
        st.write('I was not clicked 😞')
