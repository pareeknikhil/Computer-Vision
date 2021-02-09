# based on code from https://www.tensorflow.org/tutorials

import tensorflow as tf
import numpy as np

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(1)
#The above method is deprecated for tensorflow 2.0.Please use the line below for tensorflow 2.0.
# tf.random.set_seed(1)

# specify path to training data and testing data

folderbig = "big"
foldersmall = "small"

train_x_location = foldersmall + "/" + "x_train.csv"
train_y_location = foldersmall + "/" + "y_train.csv"
test_x_location = folderbig + "/" + "x_test.csv"
test_y_location = folderbig + "/" + "y_test.csv"

print("Reading training data")
x_train_2d = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")
x_train_3d = x_train_2d.reshape(-1,28,28,1)
x_train = x_train_3d
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

print("Pre processing x of training data")
x_train = x_train / 255.0

# define the training model
model = tf.keras.models.Sequential([
    tf.keras.layers.MaxPool2D(4, 4, input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(8, (3,3), padding='same', activation=tf.nn.relu,strides=1,
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l1(1e-4)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,strides=1,
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l1(1e-4)),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu,strides=2,
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l1(1e-4)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.AveragePooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("train")
model.fit(x_train, y_train, epochs=200,shuffle=True,batch_size=512)

print("Reading testing data")
x_test_2d = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
x_test_3d = x_test_2d.reshape(-1,28,28,1)
x_test = x_test_3d
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

print("Pre processing testing data")
x_test = x_test / 255.0

print("evaluate")
model.evaluate(x_test, y_test)