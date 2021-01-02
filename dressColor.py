# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:44:51 2021

@author: Poorna
"""
from scipy import stats
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

def dresscolor():
    model = load_model("color_model.h5")
    img_size=224    
    prediction_dir = "C:/Users/Poorna/Desktop/Photo"
    predict_images = []
    prediction_dir = prediction_dir + '/'
    for file in os.listdir(prediction_dir):
        img = image.load_img(prediction_dir + str(file), target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = img/255
        predict_images.append(img)
    
    predict_X = np.array(predict_images)
    
    
    prediction_final = model.predict(predict_X)
    m = stats.mode(prediction_final)
    n= stats.mode(m[1],axis = None)
    return n[0]
