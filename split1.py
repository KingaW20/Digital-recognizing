import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tools import visualize_history
import time
import numpy as np

name = "MODEL1"

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_size, val_size, test_size = 0.8, 0.1, 0.1

# split to train, val, test
x_train1, x_test1, y_train1, y_test1 = \
    train_test_split(train_x, train_y, test_size=test_size, shuffle=True, stratify=train_y)
x_train1, x_val1, y_train1, y_val1 = \
    train_test_split(x_train1, y_train1, test_size=val_size/(train_size+val_size), shuffle=True, stratify=y_train1)

# saving datasets
np.save(f'{name}/x_train.npy', x_train1, allow_pickle=True)
np.save(f'{name}/y_train.npy', y_train1, allow_pickle=True)
np.save(f'{name}/x_test.npy', x_test1, allow_pickle=True)
np.save(f'{name}/y_test.npy', y_test1, allow_pickle=True)
np.save(f'{name}/x_val.npy', x_val1, allow_pickle=True)
np.save(f'{name}/y_val.npy', y_val1, allow_pickle=True)

# creating model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  # it converts an N-dimensional layer to a 1D layer

model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(f"{name}/{name}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
start_time = time.time()
history = model.fit(x_train1, y_train1, epochs=50, validation_data=(x_val1, y_val1), callbacks=[checkpoint])
end_time = time.time()
print(f"\nTime: {(end_time - start_time):.2f} s")
visualize_history(history, name)
