import streamlit as st
import numpy as np
import requests
from PIL import Image
import io

'''
# DEEPFAKE HUNTER APPLICATION

'''

url = 'http://127.0.0.1:8000/predict'

uploaded_files = st.file_uploader("Choose photos to upload", accept_multiple_files=False, type=['png', 'jpeg', 'jpg'])

col1, col2,col3 = st.columns([1,5,1])
with col1:
    if st.button('Afficher'):

        contents = uploaded_files.read()
        pil_image = Image.open(io.BytesIO(contents))
        image = np.array(pil_image)
        st.image(image, channels="BGR")

with col2:
    if st.button('Prédire'):

        files={'image' : uploaded_files}

        response = requests.post(url, files=files)
        if response.status_code == 200:
            wb = response.json()
            if wb['result'] < 0.5:
                real = '<p style="font-family:sans-serif; color:Green; font-size: 20px;">Real</p>'
                st.markdown(real, unsafe_allow_html=True)
            else:
                fake = '<p style="font-family:sans-serif; color:Red; font-size: 20px;">Fake</p>'
                st.markdown(fake, unsafe_allow_html=True)
        else:
            st.write('Error ', str(response.status_code))
