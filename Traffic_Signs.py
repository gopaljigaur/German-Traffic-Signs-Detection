import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import Dense, Conv2D,MaxPool2D,Flatten,Dropout
from keras.models import Sequential,load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import cv2

data=[]
labels=[]

for i in range(43) :
    path = "Train/{0}/".format(i)
    print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((32,32))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print("error")
            
imgs=np.array(data)
labels=np.array(labels)

s=np.arange(imgs.shape[0])
np.random.seed(43)
np.random.shuffle(s)
imgs=imgs[s]
labels=labels[s]

val_split=0.2

(X_train,X_val)=imgs[(int)(val_split*len(labels)):],imgs[:(int)(val_split*len(labels))]
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255
(y_train,y_val)=labels[(int)(val_split*len(labels)):],labels[:(int)(val_split*len(labels))]

from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

model = Sequential()
model.add(Conv2D(32,(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(64,(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_val, y_val))

#graphs

plt.figure(0)
plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

model.save('traffic-signs.h5')
