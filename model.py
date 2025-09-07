import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#loading the training data from keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#splitting the training set into validation and smaller training set
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255

#creating the sequential neural network architecture: input layer, hidden layer with ReLU activation, 
#another hidden layer with ReLU activation, and fibally a softmax layer since this is a multi-classification problem
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

#setting a seed so this code is replicable
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

#selecting the loss, optimizer, and metrics
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

#training the model uing 30 epochs
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

#-------------------------------------- TESTING --------------------------------------

#getting images that the network has not seen yet to see how well it can identify them
X_new = X_test[:3]
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
#I took images of an ankle boot, pullover, and trouser

y_pred = np.argmax(model.predict(X_new), axis=-1)
print(y_pred)
#the model outputed [9, 2, 1], which according to the datasets classifications,
#are the identifiers for an ankle boot, pullover and trouser
#Therefore we now know that the model is able to accurately identify
