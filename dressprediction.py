# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:54:27 2020

@author: Poorna
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import pickle
import os
import pandas as pd
from keras.preprocessing import image
from scipy import stats
from keras.datasets import fashion_mnist

dataset = fashion_mnist.load_data()

dataset

df = pd.DataFrame(dataset)
df

df.head()

df.shape

tf.__version__

fashion_mnist = keras.datasets.fashion_mnist

fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/Top','Trouser','Pull-over','Dress','Coat','Sandal','Shirt',
               'Sneker','Bag','Ankel Boot']

train_images[0]
train_labels

new_train_labels = []
new_train_images = []
new_test_labels = []
new_test_images = []
removing_data_labels = [5,7,8,9]
print(train_labels.shape[0])
for i in range(train_labels.shape[0]):
    if(train_labels[i] not in removing_data_labels):
        new_train_labels.append(train_labels[i])
        new_train_images.append(train_images[i])
for i in range(test_labels.shape[0]):
    if(test_labels[i] not in removing_data_labels):
        new_test_labels.append(test_labels[i])
        new_test_images.append(test_images[i])

plt.figure()
plt.imshow(new_train_images[1100])
plt.colorbar()

train_images = np.array(new_train_images)
train_labels = np.array(new_train_labels)
test_images = np.array(new_test_images)
test_labels = np.array(new_test_labels)

train_images = train_images/255.0
test_images = test_images/255.0

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i],cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation= tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)])

#Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics =['accuracy'])
#opt = SGD(lr=0.01, momentum=0.9)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	
#Train the model
model.fit(train_images,train_labels,epochs=10)
file_name= "finalizedModel2.sav"
#pickle.dump(model,open( file_name, "wb"))
model.save('model_weight.h5')
#loaded_model = pickle.load(open(file_name,"rb"))
#model.save('saved_model/my_model')
#new_model = tf.keras.models.load_model('saved_model/my_model') 
#Eavaluate accuracy
model.summary()

test_loss,test_acc = model.evaluate(test_images,test_labels,verbose=2)
print('Test accuracy: ', test_acc)

#Make prediction
prediction = model.predict(test_images)
prediction
np.argmax(prediction[0])
test_labels[0]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i],cmap = plt.cm.binary)
    predicted_label = np.argmax(prediction[i])
    true_label = test_labels[i]
    color = ""
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    
    plt.xlabel("{}({})".format(class_names[predicted_label],class_names[true_label],color=color))

img = test_images[0]
print(img)

#Add the image to a batch where it's the only member
img = (np.expand_dims(img,0))
print(img.shape)

#Predict the image
prediction = model.predict(img)
print(prediction)
prediction = prediction[0]
np.argmax(prediction)

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


#if __name__ == "__main__":

    # load model
    #model = load_model("model_aug.h5")
    
prediction_dir = "C:/Users/Poorna/Desktop/Photo"
predict_images = []
img_size = 28
prediction_dir = prediction_dir + '/'
for file in os.listdir(prediction_dir):
    print(file)
    img = image.load_img(prediction_dir + str(file), target_size=(img_size, img_size))
    img = image.img_to_array(img)
    img = img/255
    import cv2
for file in os.listdir(prediction_dir):
    img = cv2.imread(prediction_dir + str(file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(28,28))
    img = np.reshape(img,[1,28,28])
    predict_images.append(img)
    print(img.shape)
    
predict_X = np.array(predict_images)
print("shape",predict_X.shape)

prediction_final = []
for i in predict_X:
    prediction_final.append(model.predict(i))


m = stats.mode(prediction_final)
print(m)

   

# =============================================================================
# img_size = 28
# prediction_dir = "C:/Users/Poorna/Desktop/Photo/"
# predict_images = []
# prediction_dir = prediction_dir + '/'
# for file in os.listdir(prediction_dir):
#     print(file)
#     img = image.load_img(prediction_dir + str(file), target_size=(img_size, img_size))
#     img = image.img_to_array(img)
#     img = img/255
#     #img = model.fit.resize(img_size, 12, 22)
#     predict_images.append(img)
#     print(img.shape)
#     
# predict_X = np.array(predict_images)
# 
# prediction_final = model.predict(predict_X)
# print(prediction_final)
# m = stats.mode(prediction_final)
# print(m)
# =============================================================================


