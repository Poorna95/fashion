# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:14:07 2020

@author: Poorna
"""
import tensorflow as tf
from keras.models import Sequential,Model,load_model
from keras.optimizers import SGD
from keras.layers import BatchNormalization, Lambda, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.merge import Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
import keras.backend as K

tf.__version__


def beer_net(num_classes):
    # placeholder for input image
    input_image = Input(shape=(224,224,3))
    # ============================================= TOP BRANCH ===================================================
    # first top convolution layer
    top_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),
                              input_shape=(224,224,3),activation='relu')(input_image)
    top_conv1 = BatchNormalization()(top_conv1)
    top_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_conv1)

    # second top convolution layer
    # split feature map by half
    top_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(top_conv1)
    top_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(top_conv1)

    top_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv2)
    top_top_conv2 = BatchNormalization()(top_top_conv2)
    top_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv2)

    top_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv2)
    top_bot_conv2 = BatchNormalization()(top_bot_conv2)
    top_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv2)

    # third top convolution layer
    # concat 2 feature map
    top_conv3 = Concatenate()([top_top_conv2,top_bot_conv2])
    top_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_conv3)

    # fourth top convolution layer
    # split feature map by half
    top_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(top_conv3)
    top_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(top_conv3)

    top_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
    top_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)

    # fifth top convolution layer
    top_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_top_conv4)
    top_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv5) 

    top_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(top_bot_conv4)
    top_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv5)

    # ============================================= TOP BOTTOM ===================================================
    # first bottom convolution layer
    bottom_conv1 = Convolution2D(filters=48,kernel_size=(11,11),strides=(4,4),
                              input_shape=(224,224,3),activation='relu')(input_image)
    bottom_conv1 = BatchNormalization()(bottom_conv1)
    bottom_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_conv1)

    # second bottom convolution layer
    # split feature map by half
    bottom_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(bottom_conv1)
    bottom_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(bottom_conv1)

    bottom_top_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv2)
    bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
    bottom_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv2)

    bottom_bot_conv2 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv2)
    bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
    bottom_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv2)

    # third bottom convolution layer
    # concat 2 feature map
    bottom_conv3 = Concatenate()([bottom_top_conv2,bottom_bot_conv2])
    bottom_conv3 = Convolution2D(filters=192,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_conv3)

    # fourth bottom convolution layer
    # split feature map by half
    bottom_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(bottom_conv3)
    bottom_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(bottom_conv3)

    bottom_top_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
    bottom_bot_conv4 = Convolution2D(filters=96,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)

    # fifth bottom convolution layer
    bottom_top_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_top_conv4)
    bottom_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv5) 

    bottom_bot_conv5 = Convolution2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same')(bottom_bot_conv4)
    bottom_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv5)

    # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
    conv_output = Concatenate()([top_top_conv5,top_bot_conv5,bottom_top_conv5,bottom_bot_conv5])

    # Flatten
    flatten = Flatten()(conv_output)

    # Fully-connected layer
    FC_1 = Dense(units=4096, activation='relu')(flatten)
    FC_1 = Dropout(0.25)(FC_1)
    FC_2 = Dense(units=4096, activation='relu')(FC_1)
    FC_2 = Dropout(0.25)(FC_2)
    output = Dense(units=num_classes, activation='softmax')(FC_2)
    
    model = Model(inputs=input_image,outputs=output)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

img_rows , img_cols = 224,224
num_classes = 11
batch_size = 32
nb_epoch = 5

# initialise model
model = beer_net(num_classes)

filepath = 'color_weights.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#Black_PATH = 'C:/Users/Poorna/Desktop/color/Black'
#Blue_PATH = 'C:/Users/Poorna/Desktop/color/Blue'
#Brown_PATH = 'C:/Users/Poorna/Desktop/color/Brown'
#Green_PATH = 'C:/Users/Poorna/Desktop/color/Green'
#Grey_PATH = 'C:/Users/Poorna/Desktop/color/Grey'
#Orange_PATH = 'C:/Users/Poorna/Desktop/color/Orange'
#Pink_PATH = 'C:/Users/Poorna/Desktop/color/Pink'
#Purple_PATH = 'C:/Users/Poorna/Desktop/color/Purple'
#Red_PATH = 'C:/Users/Poorna/Desktop/color/Red'
#White_PATH = 'C:/Users/Poorna/Desktop/color/White'
#Yellow_PATH = 'C:/Users/Poorna/Desktop/color/Yellow'

#DIRS = [(0, Black_PATH), (1, Blue_PATH),(2, Yellow_PATH),(3, Brown_PATH),(4, Green_PATH),(5, Grey_PATH),
#           (6, Orange_PATH),(7, Pink_PATH),(8, Purple_PATH),(9,Red_PATH),(10,White_PATH)]

training_set = train_datagen.flow_from_directory(
            'C:/Users/Poorna/Desktop/color/train',
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

test_set = test_datagen.flow_from_directory(
            'C:/Users/Poorna/Desktop/color/test',
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

history = model.fit_generator(
        training_set,
        steps_per_epoch=1000,
        epochs=nb_epoch,
        validation_data=test_set,
        validation_steps=300,
        callbacks=callbacks_list)

model.save('color_model.h5')

#load model
model = load_model("color_model.h5")

model.summary()

import matplotlib.pyplot as plt
from keras.preprocessing import image
import os
img_size=224

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predictions = model.predict(training_set)
#print(predictions)
#print(confusion_matrix(predictions.argmax(axis=1), test_set.argmax(axis=1)))

#from sklearn.metrics import classification_report
#print(classification_report(predictions, test_set))

import random
random_indices = [random.randint(0, 280) for i in range(9)]

plt.figure(figsize=(10,10))
#for i, index in enumerate(random_indices):
 #   pred = predictions[index]
  #  pred = '0' if pred==0 else '1'
  #  actual = '0' if test_set[index]==0 else '1'
  #  plt.subplot(3,3,i+1)
  #  plt.imshow(training_set[index], cmap='gray', interpolation='none')
  #  plt.title(f"Predicted: {pred}, \n Class: {actual}")
  #  plt.tight_layout()
    
prediction_dir = "C:/Users/Poorna/Desktop/Photo"
predict_images = []
prediction_dir = prediction_dir + '/'
for file in os.listdir(prediction_dir):
    print(file)
    img = image.load_img(prediction_dir + str(file), target_size=(img_size, img_size))
    img = image.img_to_array(img)
    img = img/255
    predict_images.append(img)

predict_X = np.array(predict_images)

from scipy import stats
prediction_final = model.predict(predict_X)
print(prediction_final)
m = stats.mode(prediction_final)
print(m)
print(m[1])
n= stats.mode(m[1],axis = None)
print(n[0])
