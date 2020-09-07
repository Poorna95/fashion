# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Tensorflow  and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
#from keras.datasets import fashion_mnist

tf.__version__

#Import the fashion MINIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/Top','Trouser','Pull-over','Dress','Coat','Sandal','Shirt',
               'Sneker','Bag','Ankel Boot']

#Explore the data
train_images.shape
len(train_labels)

#Preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca.grid(False)

train_images = train_images/255.0
test_images = test_images/255.0

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i],cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
#Setup the layers
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

#Eavaluate accuracy
test_loss,test_acc = model.evaluate(test_images,test_labels,verbose=2)
print('Test accuracy: ', test_acc)

#Make prediction
prediction = model.predict(test_images)
prediction
np.argmax(prediction[0])
test_labels[0]

#Color correct prediction in green,incorrect prediction in red
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
    
    plt.xlabel("{}({})".format(class_names[predicted_label],
                               class_names[true_label],color=color)
               
#Grab an image from the test dataset
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



