import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_data(name):
    x_train = np.load(f'{name}/x_train.npy', allow_pickle=True)
    y_train = np.load(f'{name}/y_train.npy', allow_pickle=True)
    x_test = np.load(f'{name}/x_test.npy', allow_pickle=True)
    y_test = np.load(f'{name}/y_test.npy', allow_pickle=True)
    if name != "MODEL3":
        x_val = np.load(f'{name}/x_val.npy', allow_pickle=True)
        y_val = np.load(f'{name}/y_val.npy', allow_pickle=True)
    else:
        x_val = None
        y_val = None
    return x_train, y_train, x_test, y_test, x_val, y_val


def evaluate_model_part(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    print("Dokładność: {:.4f}".format(accuracy))
    print("Precyzja: {:.4f}".format(precision))
    print("Pełność: {:.4f}".format(recall))
    print("F-miara: {:.4f}".format(f1))


def evaluate_model(name):
    print(name)
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(name)

    # load model
    model = tf.keras.models.load_model(f"{name}/{name}.h5")
    y_train_pred = model.predict(x_train)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    y_test_pred = model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    if name != "MODEL3":
        y_val_pred = model.predict(x_val)
        y_val_pred = np.argmax(y_val_pred, axis=1)

    # model evaluation
    print("TRAINING DATA")
    evaluate_model_part(y_train, y_train_pred)
    print("\nTESTING DATA")
    evaluate_model_part(y_test, y_test_pred)
    if name != "MODEL3":
        print("\nVALIDATION DATA")
        evaluate_model_part(y_val, y_val_pred)


# SPLIT 1
evaluate_model("MODEL1")
print("\n\n")

# SPLIT 2
evaluate_model("MODEL2")
print("\n\n")

# SPLIT 2
evaluate_model("MODEL3")
print("\n\n")
