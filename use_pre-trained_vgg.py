# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:52:02 2021

@author: Ahmed Fayed
"""

from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


# re-size all the images to this
Image_Size = [160, 160]


# training config
epochs = 5
batch_size = 32


train_path = 'E:/Software/Kaggle/fruits-360/Training'
valid_path = 'E:/Software/Kaggle/fruits-360/Test'


# useful for getting number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')


# useful for getting number of class
folders = glob(train_path + '/*')


plt.imshow(image.load_img(np.random.choice(image_files)))
# plt.show()



# add preprocessing layer to the front of VGG
# weights: we cant start with a pure random weights if we want but we w'll use the defult weights 'imagenet'
# include_top = flase: since we want everything except that the last layer of VGG since we'll be training our own final classifier
vgg = VGG16(input_shape=Image_Size + [3], weights='imagenet', include_top=False)


# don't train existing wights
for layer in vgg.layers:
    layer.trainable = False
    
    
# creating our own layers using keras
x = Flatten()(vgg.output)

# we only create one hidden layer since we're creating a logistic regression
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(vgg.input, prediction)


# View the structure of the model
model.summary()



model.compile(
    loss='categorical_crossentropy', 
    optimizer='rmsprop', 
    metrics=['accuracy'])


# creating instance of ImageDataGenerator
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input)


test_gen = gen.flow_from_directory(valid_path, target_size=Image_Size)
print(test_gen.class_indices)

labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
    labels[v] = k
    

# should be a strangley colored image due to VGG weights being BGR
for x, y in test_gen:
    print('min: ',x[0].min(), '  max: ', x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break


train_generator = gen.flow_from_directory(
    train_path,
    target_size=Image_Size,
    shuffle=True,
    batch_size=batch_size)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=Image_Size,
    shuffle=True,
    batch_size=batch_size)


# fit the model
r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    steps_per_epoch=len(image_files) // batch_size,
    validation_steps=len(valid_image_files)// batch_size)



def get_confusion_matrix(data_path, N):
    
    i = 0
    predictions = []
    targets = []
    
    for x,y in gen.flow_from_directory(data_path, target_size=Image_Size, shuffle = False, batch_size= batch_size * 2):
        
        i += 0
        if i%50 == 0:
            print(i)
        
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        
        if len(targets) >= N:
            break
        
        cm = confusion_matrix(targets, predictions)
        return cm
        
    
cm = get_confusion_matrix(train_path, len(image_files))
print(cm)

valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)



# ploting some data

#loss
plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()


# Accuracy
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()


from util import plot_confusion_matrix

plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')































