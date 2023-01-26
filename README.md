We coded a cool Deepfake detector app. Feed it an image and know if the image is real of if it has been altered by deepfake technology.

Scripts for the Dataset creation and the Deep Learning model initialization and training are situated in the ml_logic folder.

To run the app, create the API endpoint with the api.py script and a Uvicorn server and call it through the simple streamlit app located in the separate app_deepfake repo.

We used Python as a language, Tensoflow to build our model, Fast API for the API and Streamlit for the simple web app. Our accuracy is 75%.

Raw data can be downloaded here: https://www.kaggle.com/datasets/dagnelies/deepfake-faces
