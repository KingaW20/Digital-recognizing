import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from tools import evaluate_model_part, visualize_measures, evaluate_model, visualize_history
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def evaluate_models(x_test, y_test):
    measures = ["accuracy", "precision", "recall"]
    print("model", "\t", "accuracy", "\t", "precision", "\t", "recall")
    for measure in measures:
        model = tf.keras.models.load_model(f"improvement/{measure}.h5")
        y_test_pred = model.predict(x_test)
        accuracy, precision, recall = evaluate_model_part(y_test, y_test_pred, improvement=True)
        print(measure, "\t", accuracy, "\t", precision, "\t", recall)


def create_model(layer_number=3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(layer_number):
        model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


name = "MODEL2"
# load data
x_train = np.load(f'{name}/x_train.npy', allow_pickle=True)
y_train = np.load(f'{name}/y_train.npy', allow_pickle=True)
x_test = np.load(f'{name}/x_test.npy', allow_pickle=True)
y_test = np.load(f'{name}/y_test.npy', allow_pickle=True)
x_val = np.load(f'{name}/x_val.npy', allow_pickle=True)
y_val = np.load(f'{name}/y_val.npy', allow_pickle=True)

# hyperparameter optimization
models = [0] * 3
models[0] = create_model(1)
models[1] = create_model(2)
models[2] = create_model(3)

# training
i = 1
for model in models:
    layer_name = "layer" + str(i)
    checkpoint = \
        ModelCheckpoint(f"improvement/{layer_name}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(x_train, y_train, epochs=25, validation_data=(x_val, y_val), callbacks=[checkpoint])
    visualize_history(history, name, layer_name)
    evaluate_model(name, layer_name)
    i += 1
    print("-------------------------------------------------------\n")
