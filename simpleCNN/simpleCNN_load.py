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