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

uploaded_files = st.file_uploader("Choose a picture to upload", accept_multiple_files=False, type=['png', 'jpeg', 'jpg'])
st.set_option('deprecation.showfileUploaderEncoding', False) # Enabling the automatic file decoder

if st.button('Show the picture'):

    image = Image.open(uploaded_files)

    fig = plt.figure(2, figsize=(200, 200))
    fig, ax = plt.subplots()


    st.image(image, width=500)



url = 'http://127.0.0.1:8000/predict'
if url == 'http://127.0.0.1:8000/predict':

    st.markdown('Please click on the "Analysis" button to submit your picture to activate the DeepFake Detection')

if st.button('Analysis'):

    files={'image' : uploaded_files}

    response = requests.post(url, files=files)
    if response.status_code == 200:
        wb = response.json()
        # wb = response.text
        # st.write(wb + '  ' + str(response.status_code)
        # st.write('the prediction  is ', wb['result'])
        if wb["result"] >= 0.5:
            st.write("DEEPFAKE ALERT: there is a 73% chance this picture was altered by deepfake technology")
        else:
            st.write("You are safe: this picture looks real to us!")

    else:
        st.write('the response.status_code  is ', str(response.status_code) + '   ' + response.text)
        st.write('the response.status_code  is ', str(response.status_code) + '   ' + uploaded_files.name)
else:
        st.write('Waiting for an image to process')
