from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
import numpy as np
from tools import normalize_data, visualize_history, evaluate_model
from hard_data_generator import generate
import tensorflow as tf


def get_images(data, x_train, y_train):
    data_x = [x_train[indeks] for indeks in data]
    data_x = np.array(data_x, dtype=np.float32)
    data_y = [y_train[indeks] for indeks in data]
    data_y = np.array(data_y, dtype=np.float32)
    return data_x, data_y


def train(load_model_name, model_name, data_x, data_y, x_val, y_val):
    model = tf.keras.models.load_model(f"{load_model_name}/{load_model_name}.h5")
    checkpoint = ModelCheckpoint(f"improvement/{model_name}.h5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    history = model.fit(data_x, data_y, epochs=50, validation_data=(x_val, y_val), callbacks=[checkpoint])
    visualize_history(history, name, model_name)
    evaluate_model(name, model_name)


# normalization and augmentation
szum, hard_examples, all = generate()
name = "MODEL2"
(x_train, y_train), (test_x, test_y) = mnist.load_data()
x_train = normalize_data(x_train)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.

# prepare hard examples
szum_x, szum_y = get_images(szum, x_train, y_train)
hard_examples_x, hard_examples_y = get_images(hard_examples, x_train, y_train)
all_x, all_y = get_images(all, x_train, y_train)
print(len(szum_x))
print(len(hard_examples_x))
print(len(all_x))
print("\n\n")

# load sets
x_train = np.load(f'{name}/x_train.npy', allow_pickle=True)
y_train = np.load(f'{name}/y_train.npy', allow_pickle=True)
x_test = np.load(f'{name}/x_test.npy', allow_pickle=True)
y_test = np.load(f'{name}/y_test.npy', allow_pickle=True)
x_val = np.load(f'{name}/x_val.npy', allow_pickle=True)
y_val = np.load(f'{name}/y_val.npy', allow_pickle=True)

# training
train("MODEL2", "szum", szum_x, szum_y, x_val, y_val)
print("------------------------------------------------------")
train("MODEL2", "hard-examples", hard_examples_x, hard_examples_y, x_val, y_val)
print("------------------------------------------------------")
train("MODEL2", "all", all_x, all_y, x_val, y_val)
