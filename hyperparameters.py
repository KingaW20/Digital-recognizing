import numpy as np
import tensorflow as tf
from tools import evaluate_model_part, visualize_measures, evaluate_model
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


def create_model(neurons=32, activation='relu'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dense(2 * neurons, activation=activation))
    model.add(tf.keras.layers.Dense(neurons, activation=activation))
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
model = KerasClassifier(build_fn=create_model)
param_grid = {'neurons': [32, 64, 128, 256, 512], 'activation': ['relu', 'sigmoid']}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train, validation_data=(x_val, y_val))
best_params = grid_search.best_params_
print("Najlepsze parametry:", best_params)
model = create_model(neurons=best_params['neurons'], activation=best_params['activation'])

# training
num_epochs = 25
train_accuracy, train_precision, train_recall = [0] * num_epochs, [0] * num_epochs, [0] * num_epochs
val_accuracy, val_precision, val_recall = [0] * num_epochs, [0] * num_epochs, [0] * num_epochs
accuracy_best_epoch, precision_best_epoch, recall_best_epoch = 0, 0, 0

for epoch in range(num_epochs):
    print("Epoka ", epoch+1)
    model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

    y_train_pred = model.predict(x_train)
    train_accuracy[epoch], train_precision[epoch], train_recall[epoch] = \
        evaluate_model_part(y_train, y_train_pred, improvement=True)

    y_val_pred = model.predict(x_val)
    val_accuracy[epoch], val_precision[epoch], val_recall[epoch] = \
        evaluate_model_part(y_val, y_val_pred, improvement=True)

    if val_accuracy[epoch] == max(val_accuracy):
        model.save(f'improvement/accuracy.h5')
        accuracy_best_epoch = epoch
    if val_precision[epoch] == max(val_precision):
        model.save(f'improvement/precision.h5')
        precision_best_epoch = epoch
    if val_recall[epoch] == max(val_recall):
        model.save(f'improvement/recall.h5')
        recall_best_epoch = epoch

print("Epoka ", accuracy_best_epoch, "DOKŁADNOŚĆ - ZAPISANO\t", max(val_accuracy))
print("Epoka ", precision_best_epoch, "PRECYZJA - ZAPISANO\t", max(val_precision))
print("Epoka ", recall_best_epoch, "PEŁNOŚĆ - ZAPISANO\t", max(val_recall))

# visualize measures
visualize_measures(train_accuracy, train_precision, train_recall, title="Zbiór trenujący")
visualize_measures(val_accuracy, val_precision, val_recall, title="Zbiór walidacyjny")

evaluate_model(name)
