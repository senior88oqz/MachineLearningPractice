import numpy as np
import matplotlib.pyplot as plt

data = np.load("./data.npz")
X = data['train_X']
y = data['train_y']
X_test = data['test_X']

X_train = X
y_train = y

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# reshape to be [samples][depth/pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64).astype('float32')    #grey-scale -> depth = 1; RBG -> depth =3
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
# convert 1d class arrays to 10d class matrix
y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]

def save_model():
# serialize model to JSON
    model_json = model.to_json()
    with open("CNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("CNN.h5")
    print("Saved model to disk")


# construct CNN model
def set_model():
    # create model
    model = Sequential()
    #first layer
    model.add(Conv2D(30, (4, 4), input_shape=(1, 64, 64), activation='relu'))  # Conv2D(a,b,c)-> a: number of convolution filter,
    																		                     #b/c: number of column rows/columns in each convolutional kernel
    																		   #input_shape-> shape of 1 sampel (e.g. X_train.shape)
    #second layer
    model.add(Conv2D(60, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    #third layer
    model.add(Conv2D(120, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    #Fully connected Dense layer
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = set_model()

def model_train():
    # Fit the model
    #model.fit(X_train, y_train, validation_data=(X_heldout, y_heldout), epochs=1, batch_size=1000, verbose=2)
    model.fit(X_train, y_train, epochs=1, batch_size=1000, verbose=2)
    save_model()

for i in range (20):
    print("Total epoch= ", i + 1)
    model_train()

X_test = X_test.reshape(X_test.shape[0], 1, 64, 64).astype('float32')
X_test /= 255
predictions = model.predict(X_test, batch_size=500, verbose=0)
predict = predictions.argmax(1)

import csv
f = open('CNN_predict.csv', 'w')
writer = csv.writer(f)
index = np.linspace(1, 500, 500)
writer.writerow(index)
writer.writerow(predict)
