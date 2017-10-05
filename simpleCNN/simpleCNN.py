import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = np.load("./data.npz")
X = data['train_X']
y = data['train_y']
X_test = data['test_X']

X_train,X_heldout,y_train,y_heldout = train_test_split(X,y,test_size=0.3, random_state = 1)
X_train=X
y_train=y

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64).astype('float32')
X_heldout = X_heldout.reshape(X_heldout.shape[0], 1, 64, 64).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_heldout = X_heldout / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_heldout = np_utils.to_categorical(y_heldout)
num_classes = y_heldout.shape[1]


def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_heldout, y_heldout), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_heldout, y_heldout, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# serialize model to JSON
model_json = model.to_json()
with open("simpleCNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("simpleCNN.h5")
print("Saved model to disk")

from keras.models import model_from_json
json_file = open('simpleCNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("simpleCNN.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_heldout, y_heldout, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))