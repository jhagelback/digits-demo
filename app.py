from flask import Flask, request, render_template, jsonify
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
import time
import numpy as np

# Define Flask application
app = Flask(__name__)


#
# Builds and trains a CNN model for digit recognition.
#
def build_model():
    global X_test, Y_test

    # Load pre-shuffled MNIST data into train and test sets
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

    # Preprocess input data
    X_train = X_train_raw.reshape(X_train_raw.shape[0], 28, 28, 1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], 28, 28, 1)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    # Normalize values to 0...1
    X_train /= 255
    X_test /= 255

    # Preprocess class labels
    # Convert labels to 10-dimensional one-hot vectors
    Y_train = to_categorical(y_train_raw, 10)
    Y_test = to_categorical(y_test_raw, 10)

    # Start timer
    start = time.time()

    # Convolutional Neural Network model
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces size to 14x14
    model.add(Convolution2D(32, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces size to 7x7
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation="softmax"))

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train model on training data
    model.fit(X_train, Y_train, batch_size=100, epochs=10, verbose=1)

    # Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)

    # Stop timer
    end = time.time()

    # Print results
    print("Test Accuracy: {0:0.2f}%".format(score[1] * 100))
    print("Time elapsed: {0:0.2f} sec".format(end - start))

    # Save model
    # Serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


#
# Loads CNN model from file.
#
def load_model(filename):
    # Load json and create model
    json_file = open(filename + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights(filename + ".h5")
    print("Loaded model from disk")
    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


#
# Tests the loaded model.
#
def test_prediction():
    global model

    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

    # Preprocess input data
    X_test = X_test_raw.reshape(X_test_raw.shape[0], 28, 28, 1)
    X_test = X_test.astype("float32")
    # Normalize values to 0...1
    X_test /= 255

    # Preprocess class labels
    # Convert labels to 10-dimensional one-hot vectors
    Y_test = to_categorical(y_test_raw, 10)

    # Classifies some test images
    for i in range(0,3):
        X_p = np.asarray([X_test[i]])
        pred = model.predict(X_p)
        predicted_label = np.argmax(pred[0])
        print("Predicted:", predicted_label)
        print("Actual", y_test_raw[i])


#build_model()
model = load_model("model")
test_prediction()

#
# Start page.
# Find process with:
# ps aux | grep flask
#
@app.route("/")
def index():
    return render_template("index_en.html")

#
# Classifies a drawn digit.
#
@app.route("/classify")
def classify():
    global model

    # Get data as param
    if "data" in request.args:
        data = request.args["data"]

    # Convert query param to numpy matrix
    X_0 = np.zeros(shape=(28, 28, 1))
    for i in range(0,784):
        v = data[i:i+1]
        r = int(i / 28)
        c = int(i % 28)
        X_0[r][c] = [int(v)]

    # Convert to X input
    X_p = np.asarray([X_0])
    # Predict example
    pred = model.predict(X_p)
    # Prediction result
    p_res = pred[0]
    p_label = np.argmax(p_res)
    p_prob = p_res[p_label]

    # Result
    print("Predicted: ", p_label)
    print("Prob: ", p_prob)

    # Create response
    response = {}
    response.update({"label" : int(p_label)})
    response.update({"prob": float(p_prob)})

    probs = [0] * 10
    for i in range(0, 10):
        probs[i] = float(p_res[i])

    response.update({"probs": probs})

    return jsonify(response)

