import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tools import visualize_history
import time

#SPLIT 3
load_data_name = "MODEL2"
name = "MODEL3"
x_train1 = np.load(f'{load_data_name}/x_train.npy', allow_pickle=True)
y_train1 = np.load(f'{load_data_name}/y_train.npy', allow_pickle=True)
x_test1 = np.load(f'{load_data_name}/x_test.npy', allow_pickle=True)
y_test1 = np.load(f'{load_data_name}/y_test.npy', allow_pickle=True)
x_val1 = np.load(f'{load_data_name}/x_val.npy', allow_pickle=True)
y_val1 = np.load(f'{load_data_name}/y_val.npy', allow_pickle=True)

# split to train, test
x_train1 = np.append(x_train1, x_val1, 0)
y_train1 = np.append(y_train1, y_val1)

# saving datasets
np.save(f'{name}/x_train.npy', x_train1, allow_pickle=True)
np.save(f'{name}/y_train.npy', y_train1, allow_pickle=True)
np.save(f'{name}/x_test.npy', x_test1, allow_pickle=True)
np.save(f'{name}/y_test.npy', y_test1, allow_pickle=True)

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
print(f"\nTime: {(end_time - start_time):.2f} s\n\n")
visualize_history(history, name)
