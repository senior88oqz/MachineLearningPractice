import numpy as np

data = np.load("./data.npz")
X_test = data['test_X']


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
X_test = X_test.reshape(X_test.shape[0], 1, 64, 64).astype('float32')
X_test /= 255

from keras.models import model_from_json
json_file = open('./simpleCNN/simpleCNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./simpleCNN/simpleCNN.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

predictions = loaded_model.predict(X_test, batch_size=500, verbose=0)
predict1 = predictions.argmax(1)
predict2 = predictions.argmin(1)


# score = loaded_model.evaluate(X_heldout, y_heldout, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

import csv
# csvfile = file('prediction2.csv', 'wb')
f =open('predict2.csv','w')
writer = csv.writer(f)
index = np.linspace(1,500,500)
writer.writerow(index)
writer.writerow(predict2)
