# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:10:33 2021

@author: Poorna
"""
import numpy as np

from keras.preprocessing import image
import os
import pickle
import glob


img_size = 300

def removeMP4():
    for i in glob.glob(os.path.join("C:/Users/Poorna/Desktop/Photo","*.mp4")):
        try:
           os.chmod(i,0o777)
           os.remove(i)
        except OSError:
           pass


def categoriesMaleFemale():
    file_name= "finalizedModel1.sav"
    loaded_model = pickle.load(open(file_name,"rb"))

    prediction_dir = "C:/Users/Poorna/Desktop/Photo"
    predict_images = []
    prediction_dir = prediction_dir + '/'
    for file in os.listdir(prediction_dir):
        img = image.load_img(prediction_dir + str(file), target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = img/255
        predict_images.append(img)

    predict_X = np.array(predict_images)
    
    prediction_final = loaded_model.predict_classes(predict_X)
    
    count = 0
    for file in os.listdir(prediction_dir):
        label = prediction_final[count]
        if(label == 0):
            os.remove(prediction_dir + str(file))
            print(file)
        count += 1