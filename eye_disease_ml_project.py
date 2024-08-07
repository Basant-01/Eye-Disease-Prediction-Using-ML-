# -*- coding: utf-8 -*-
"""EYE DISEASE ML PROJECT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yCJMDBAsFoElVqQE37Q23PtqdZsTDWJA

# CLASSIFICATION OF EYE DISEASES IN FUNDUS IMAGES
"""

from google.colab import drive
drive.mount('/content/drive')

"""USING PYTHON LIBRARIES"""

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import cv2
import random
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import os
#import mlxtend
import keras
import keras.utils
from tensorflow import keras
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Activation, Dropout

"""# Loading the Eye dataset"""

data = pd.read_csv("/content/drive/MyDrive/Kaggle-20240408T083538Z-001/Kaggle/full_df.csv")

data.describe

data.columns

data.head(10)

data["Patient Sex"].value_counts()

data.info()

data.shape

"""# TYPES OF DISEASE"""

data['Left-Diagnostic Keywords'].value_counts()

data['Right-Diagnostic Keywords'].value_counts()



"""# Extracting Drusen & Normal information from the Dataset"""

def has_drusen(text):
    if "drusen" in text:
        return 1
    else:
        return 0

data["left_drusen"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_drusen(x))
data["right_drusen"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_drusen(x))

data["left_drusen"][0:10]

data["right_drusen"][0:10]

left_drusen = data.loc[data["Left-Diagnostic Keywords"] =="drusen"]["Left-Fundus"].values
left_drusen[:30]

right_drusen = data.loc[data["Right-Diagnostic Keywords"] =="drusen"]["Right-Fundus"].values
right_drusen[:30]

print("Number of images in left drusen: {}".format(len(left_drusen)))
print("Number of images in right drusen: {}".format(len(right_drusen)))

left_normal = data.loc[(data.D ==0) & (data["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(250,random_state=42).values
left_normal[:15]

right_normal = data.loc[(data.D ==0) & (data["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(250,random_state=42).values# In[ ]:
right_normal[:15]

drusen = np.concatenate((left_drusen,right_drusen),axis=0)
normal = np.concatenate((left_normal,right_normal),axis=0)

print(len(drusen),len(normal))

file_names = []
labels = []

for text, label, file_name in zip(data["Left-Diagnostic Keywords"], data["C"], data["Left-Fundus"]):

    if(("drusen" in text) and (label == 1)):
        file_names.append(file_name)
        labels.append(1)

    elif(("normal fundus" in text) and (label == 0)):
        file_names.append(file_name)
        labels.append(0)

for text, label, file_name in zip(data["Right-Diagnostic Keywords"], data["C"], data["Right-Fundus"]):

    if(("drusen" in text) and (label == 1)):
        file_names.append(file_name)
        labels.append(1)

    elif(("normal fundus" in text) and (label == 0)):
        file_names.append(file_name)
        labels.append(0)

print(len(file_names), len(labels))

plt.bar([0,1], [len([i for i in labels if i == 1]), len([i for i in labels if i == 0])], color = ['r', 'g'])
plt.xticks([0, 1], ['drusen', 'Normal'])
plt.show()

"""# Extracting Cataract & Normal information from the Dataset"""

def has_cataract(text):
    if "cataract" in text:
        return 1
    else:
        return 0

data["left_cataract"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
data["right_cataract"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_cataract(x))

data["left_cataract"][0:10]

data["right_cataract"][0:10]

left_cataract = data.loc[(data.C ==1) & (data.left_cataract == 1)]["Left-Fundus"].values

right_cataract = data.loc[(data.C ==1) & (data.right_cataract == 1)]["Right-Fundus"].values
right_cataract[:15]

print("Number of images in left cataract: {}".format(len(left_cataract)))
print("Number of images in right cataract: {}".format(len(right_cataract)))

left_normal = data.loc[(data.C ==0) & (data["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(250,random_state=42).values
right_normal = data.loc[(data.C ==0) & (data["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(250,random_state=42).values
right_normal[:15]

cataract = np.concatenate((left_cataract,right_cataract),axis=0)
normal = np.concatenate((left_normal,right_normal),axis=0)

print(len(cataract),len(normal))

file_names = []
labels = []

for text, label, file_name in zip(data["Left-Diagnostic Keywords"], data["C"], data["Left-Fundus"]):

    if(("cataract" in text) and (label == 1)):
        file_names.append(file_name)
        labels.append(1)

    elif(("normal fundus" in text) and (label == 0)):
        file_names.append(file_name)
        labels.append(0)

for text, label, file_name in zip(data["Right-Diagnostic Keywords"], data["C"], data["Right-Fundus"]):

    if(("cataract" in text) and (label == 1)):
        file_names.append(file_name)
        labels.append(1)

    elif(("normal fundus" in text) and (label == 0)):
        file_names.append(file_name)
        labels.append(0)

print(len(file_names), len(labels))

plt.bar([0,1], [len([i for i in labels if i == 1]), len([i for i in labels if i == 0])], color = ['r', 'g'])
plt.xticks([0, 1], ['Cataract', 'Normal'])
plt.show()

"""# Creating Dataset from images"""

from tensorflow.keras.preprocessing.image import load_img,img_to_array
dataset_dir = "preprocessed_images"
image_size=224
labels = []
dataset = []
def create_dataset(image_category,label):
    for img in tqdm(image_category):
        image_path = os.path.join(dataset_dir,img)
        try:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size,image_size))

        except:
            continue

        dataset.append([np.array(image),np.array(label)])
    random.shuffle(dataset)
    return dataset

dataset = create_dataset(drusen,1)
dataset = create_dataset(normal,0)

len(dataset)

dataset = create_dataset(cataract,1)
dataset = create_dataset(normal,0)

len(dataset)



"""# Let's see some  drusen images"""

if len(dataset) > 0:
    sample = random.choice(range(len(dataset)))
    image = dataset[sample][0]
    category = dataset[sample][1]
    if category == 0:
        label = "Normal"
    else:
        label = "drusen"
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.xlabel(label)
    plt.tight_layout()

"""# Let's see some  Cataract images"""

if len(dataset) > 0:
    plt.figure(figsize=(15,10))
    for i in range(10):
        sample = random.choice(range(len(dataset)))
        image = dataset[sample][0]
        category = dataset[sample][1]
        if category== 0:
            label = "Normal"
        else:
            label = "Cataract"

        plt.subplot(2,5,i+1)
        plt.imshow(image)
        plt.xlabel(label)
    plt.tight_layout()

"""# Dividing dataset into x(features) & y(target)"""

x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
y = np.array([i[1] for i in dataset])

x[0,224,224,3]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

x_train

from sympy.polys.domains import ZZ

#y_train>>> from sympy.polys.domains import ZZ

"""# Creating Model: Using VGG19"""

from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights="imagenet",include_top = False,input_shape=(image_size,image_size,3))

for layer in vgg.layers:
    layer.trainable = False

"""# Training The Model"""

from tensorflow.keras.utils import plot_model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
model = Sequential() # arranging the Keras layers in a sequential order
model.add(vgg)       #get a copy of an given array in 1D
model.add(Flatten()) #get a copy of an given array
model.add(Dense(1,activation="tanh"))

"""# using Softmax activation function in this model:"""

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(keras.layers.Dense(1024))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU())

model.add(keras.layers.Dense(512))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU())

model.add(keras.layers.Dense(256))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU())
model.add(Dense(1,activation="sigmoid"))

model.summary()

plot_model(model, to_file='vgg19_model.png')

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,save_freq=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)

history = model.fit(x_train,y_train,batch_size=32,epochs=1,validation_data=(x_test,y_test),
                    verbose=1,callbacks=[checkpoint,earlystop])



sns.set()
fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(history.epoch, history.history['accuracy'], label = 'train')
sns.lineplot(history.epoch, history.history['val_accuracy'], label = 'validation')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(history.epoch, history.history['loss'], label = 'train')
sns.lineplot(history.epoch, history.history['val_loss'], label = 'validation')
plt.title('Loss')
plt.tight_layout()

plt.savefig('epoch_history.png')
plt.show()

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred = (model.predict(x_test) > 0.5).astype("int32")

acc=accuracy_score(y_test,y_pred)

print("Accuracy = ",acc*100)

print(classification_report(y_test,y_pred))

print(np.shape(y_test))
# y_test = np.argmax(y_test, axis = 1)
# y_pred=y_pred.ravel()
print(np.shape(y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

model = Sequential()
model.add(Dense(24, input_dim=13, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import seaborn as sns
sns.set()
fig = plt.figure(0, (6, 4))

sns.lineplot(history.epoch, history.history['accuracy'], label = 'train')
sns.lineplot(history.epoch, history.history['val_accuracy'], label = 'validation')

plt.xlabel('Eposh')
plt.ylabel('accuracy')
plt.title('Accuracy Curve')
plt.tight_layout()

plt.savefig('Accuracy.png')
plt.show()

"""#  Prediction Drusen Data"""

plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]

    if category== 0:
        label = "Normal"
    else:
        label = "drusen"

    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "drusen"

    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout()

"""# Prediction Cataract Data"""

plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]

    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"

    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"

    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout()

model.save("my_model")

model.save('models/medical_trial_model.h5')

import tensorflow as tf
tf.keras.models.save_model(model,'model_final.hdf5')

from sklearn.metrics import balanced_accuracy_score

#define array of actual classes
actual = np.repeat([0, 0], repeats=[20, 380])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[15, 5, 5, 375])

#calculate balanced accuracy score
balanced_accuracy_score(actual, pred)

from sklearn.metrics import balanced_accuracy_score
y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]
balanced_accuracy_score(y_true, y_pred)

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, verbose=1)

model = Sequential()
model.add(Dense(24, input_dim=13, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#pip install keras
from keras.layers import Conv2D,MaxPool2D

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
nb_samples = 1000
x, y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)

print(accuracy_score(y_test, model.predict(x_test)))

model.save("my_model")

model.save('models/medical_trial_model.h5')

tf.keras.models.save_model(model,'model_final.hdf5')

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

model = Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()