import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = np.load("./data.npz")
X = data['train_X']
y = data['train_y']
X_test = data['test_X']

X_train, X_heldout, y_train, y_heldout = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = X
y_train = y

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

# reshape to be [samples][depth/pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64).astype('float32')    #grey-scale -> depth = 1; RBG -> depth =3
X_heldout = X_heldout.reshape(X_heldout.shape[0], 1, 64, 64).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_heldout = X_heldout / 255
# convert 1d class arrays to 10d class matrix
y_train = np_utils.to_categorical(y_train)
y_heldout = np_utils.to_categorical(y_heldout)
num_classes = y_heldout.shape[1]

def save_model():
# serialize model to JSON
    model_json = model.to_json()
    with open("CNN_20.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("CNN_20.h5")
    print("Saved model to disk")


# define the larger model
def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(20, (5, 5), input_shape=(1, 64, 64), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(100, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Dropout(0.2))

    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(500, activation='relu'))

    # model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = larger_model()

def model_train():
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_heldout, y_heldout), epochs=1, batch_size=1000, verbose=2)
    save_model()

for i in range (10):
    print("Total epoch= ", 21 + i)
    model_train()

# Final evaluation of the model
scores = model.evaluate(X_heldout, y_heldout, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))


X_test = X_test.reshape(X_test.shape[0], 1, 64, 64).astype('float32')
X_test /= 255
predictions = model.predict(X_test, batch_size=500, verbose=0)
predict = predictions.argmax(1)

import csv

# csvfile = file('prediction2.csv', 'wb')
f = open('predict.csv', 'w')
writer = csv.writer(f)
index = np.linspace(1, 500, 500)
writer.writerow(index)
writer.writerow(predict)
