#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,BatchNormalization,Dropout,Conv2D,MaxPool2D


# In[2]:


labels = ['NORMAL', 'KNOCKKNEES']
img_size = 150
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


# In[3]:


train = "X:/miniproject/train"
test = "X:/miniproject/test"
val = "X:/miniproject/val"


# In[4]:


train_normal = "X:/miniproject/train/normal"                           #training images dataset for normal
train_knockknees = "X:/miniproject/train/knockknees"                    #training images dataset for normal


# In[5]:

#ploting graph of training dataset 1. knockknees 2.normal
import seaborn as sns
sns.set_style('whitegrid')
sns.barplot(x=['Normal','knockknees'],y=[len(train_normal), len(train_knockknees )])


# In[6]:

#declaring input image height and width
input_height=128
input_width=128
batch_size=32


# In[7]:


train_ds=tf.keras.preprocessing.image_dataset_from_directory(train, 
                                                             color_mode='grayscale', 
                                                             image_size=(input_height, input_width), 
                                                             batch_size=batch_size)

test_ds=tf.keras.preprocessing.image_dataset_from_directory(test, 
                                                             color_mode='grayscale', 
                                                             image_size=(input_height, input_width), 
                                                             batch_size=batch_size)

val_ds=tf.keras.preprocessing.image_dataset_from_directory(val, 
                                                             color_mode='grayscale', 
                                                             image_size=(input_height, input_width), 
                                                             batch_size=batch_size)


# In[8]:




train_ds.class_names


# In[9]:

#displaying the images of training dataset by labeling 
plt.figure(figsize=(15,15))
for images, labels in train_ds.take(1):
    for i in range(25):
        
        plt.subplot(5, 5, i+1)
        plt.imshow(np.squeeze(images[i].numpy().astype('uint8')))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis('off')


# In[10]:
#image augmentation

training_dir = "X:/miniproject/train"
training_generator = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   shear_range=0.2,
                                   zoom_range=0.2 )
train_generator = training_generator.flow_from_directory(training_dir,target_size=(200,200),batch_size=4,class_mode='binary')


# In[11]:


validation_dir = "X:/miniproject/val"
validation_generator = ImageDataGenerator(rescale=1./255)
valid_generator = validation_generator.flow_from_directory(validation_dir,target_size=(200,200),batch_size=4,class_mode='binary')


# In[12]:


testing_dir = "X:/miniproject/test"
testing_generator = ImageDataGenerator(rescale=1./255)
test_generator = testing_generator.flow_from_directory(testing_dir,target_size=(200,200),batch_size=4,class_mode='binary')


# In[13]:
#building the neural networks

ak=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(200,200,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(256,(3,3),activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])


# In[14]:


ak.summary()


# In[15]:


ak.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[16]:

#training the model 
history = ak.fit_generator(train_generator,
                           validation_data = valid_generator,
                           steps_per_epoch = 70,
                           epochs = 10,
                           verbose = 1)

#attaining aron 85-90% accuracy
# In[17]:

#graph representation of train accuracy and val_accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


# In[18]:



#printing total loss & accuracy
print("Loss of the model is - " , ak.evaluate(test_generator)[0]*100 , "%")
print("Accuracy of the model is - " , ak.evaluate(test_generator)[1]*100 , "%")

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(150,150))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1,150,150, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

# load an image and predict the class
	# load the image
img = load_image('X:/miniproject/train/knockknees/image_0_2.jpeg')
# predict the class
result = ak.predict(img)
if(result[0]==1):
  print("Person suffering from Knockknees.")
else:
  print("Person not suffering from Knockknees.")
# In[24]:

#saving the model
tf.keras.models.save_model(ak, 'kk.hdf5')
#using the streamlit for deploment on web.





