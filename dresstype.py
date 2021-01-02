# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:32:14 2021

@author: Poorna
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from keras.preprocessing import image
from scipy import stats


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(28, 28))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def dresstype():
    file_name= "finalizedModel2.sav"
    model = pickle.load(open(file_name,"rb"))

    prediction_dir = "C:/Users/Poorna/Desktop/Photo"
    predict_images = []
    img_size = 28
    prediction_dir = prediction_dir + '/'
    for file in os.listdir(prediction_dir):
        img = image.load_img(prediction_dir + str(file), target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = img/255
        import cv2
    for file in os.listdir(prediction_dir):
        img = cv2.imread(prediction_dir + str(file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(28,28))
        img = np.reshape(img,[1,28,28])
        predict_images.append(img)
        
    predict_X = np.array(predict_images)
    
    prediction_final = []
    for i in predict_X:
        prediction_final.append(model.predict(i))
    
    
    m = stats.mode(prediction_final)
    return m
    
       