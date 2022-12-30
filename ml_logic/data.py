import numpy as np
import pandas as pd
# import sys
# import sklearn
# import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
# import plotly.graph_objs as go
from plotly.offline import iplot
# from matplotlib import pyplot as plt
# import os
from sklearn.model_selection import train_test_split

#Creating a dataframe of 8000 fake and 8000 real images from the source downloaded from the Kaggle link in the Readme
def get_data():
    meta = pd.read_csv('./raw_data/metadata.csv')
    real_df = meta[meta["label"] == "REAL"]
    fake_df = meta[meta["label"] == "FAKE"]
    sample_size = 8000
    real_df = real_df.sample(sample_size, random_state=42)
    fake_df = fake_df.sample(sample_size, random_state=42)
    sample_meta = pd.concat([real_df, fake_df])
    Train_set, Test_set = train_test_split(sample_meta,test_size=0.2,random_state=42,stratify=sample_meta['label'])
    Train_set, Val_set  = train_test_split(Train_set,test_size=0.3,random_state=42,stratify=Train_set['label'])
    Train_set, Pred_set = train_test_split(Train_set,test_size=0.001,random_state=42)
    y = dict()
    y[0] = []
    y[1] = []
    return Train_set, Test_set, Val_set, Pred_set

#Retreiving the actual images from the dataframe of IDs and sorting them (Fake=1 ; Real=0)
def retrieve_dataset(set_name):
    images,labels=[],[]
    for (img, imclass) in zip(set_name['videoname'], set_name['label']):
        images.append(cv2.imread('./raw_data/faces_224/'+img[:-4]+'.jpg'))
        if(imclass=='FAKE'):
            labels.append(1)
        else:
            labels.append(0)

    return np.array(images),np.array(labels)
