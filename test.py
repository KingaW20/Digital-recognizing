import tensorflow as tf
from tools import visualize_class_measures, evaluate_model_part, load_data


def evaluate_model(name):
    print(name)
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(name)

    # load model
    model = tf.keras.models.load_model(f"{name}/{name}.h5")
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    if name != "MODEL3":
        y_val_pred = model.predict(x_val)

    # model evaluation
    print('\t\t\t\t', "accuracy", "\t", "precision", "\t", "recall")
    precision_train, recall_train = evaluate_model_part(y_train, y_train_pred, "Trenujący")
    precision_test, recall_test = evaluate_model_part(y_test, y_test_pred, "Testujący")
    if name != "MODEL3":
        precision_val, recall_val = evaluate_model_part(y_val, y_val_pred, "Walidacyjny")
        visualize_class_measures(precision_train, precision_test, precision_val, name, "precyzja")
        visualize_class_measures(recall_train, recall_test, recall_val, name, "pełność")
    else:
        visualize_class_measures(precision_train, precision_test, title=name, measure="precyzja")
        visualize_class_measures(recall_train, recall_test, title=name, measure="pełność")


# SPLIT 1
evaluate_model("MODEL1")
print("\n\n")

# SPLIT 2
evaluate_model("MODEL2")
print("\n\n")

# SPLIT 2
evaluate_model("MODEL3")
