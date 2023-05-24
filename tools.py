import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score


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


def visualize_history(history: tf.keras.callbacks.History, title = "", improvement="") -> None:
    """
    Visualize history of the training model.

    Parameters
    ----------
    history : tf.keras.callbacks.History
    title : str, optional
    """
    df_hist = pd.DataFrame(history.history)

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(9)
    fig.set_figwidth(16)
    fig.suptitle(str(title) + " - " + str(improvement))

    axs[0].plot(df_hist["loss"], label="zbiór uczący")
    axs[0].plot(df_hist["val_loss"], label="zbiór walidacyjny")
    axs[0].set_title('Wartość funkcji kosztu podczas uczenia modelu')
    axs[0].set_xlabel('epoka')
    axs[0].set_ylabel('wartość')
    axs[0].legend()

    axs[1].plot(df_hist["accuracy"], label='zbiór uczący')
    axs[1].plot(df_hist["val_accuracy"], label='zbiór walidacyjny')
    axs[1].set_title('Skuteczności modelu podczas uczenia')
    axs[1].set_xlabel('epoka')
    axs[1].set_ylabel('wartość')
    axs[1].legend()

    plt.tight_layout()
    if improvement == "":
        plt.savefig(f'{title}/training.jpg')
    else:
        plt.savefig(f'improvement/{improvement}.jpg')
    plt.clf()
    # plt.show()


def normalize_data(data):
    data = data.astype(np.float32)
    for image_id in range(len(data)):
        data[image_id] = data[image_id] / 255. * 2. - 1.
    return data


def generate_data(x, y, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
    return train_datagen.flow(x, y, batch_size=batch_size)


def augment_data(x_train, y_train):
    values = [y_train.tolist().count(i) for i in range(10)]
    print(f"Categories number: {values}")

    # split images based on category
    category_x = [[] for i in range(10)]
    category_y = [[] for i in range(10)]
    for i in range(len(x_train)):
        category_x[y_train[i]].append(x_train[i])
        category_y[y_train[i]].append(y_train[i])

    # augmentation
    max_cat_number = max(values)
    last_digit = max_cat_number % 10
    new_max_count = max_cat_number + 10 - last_digit
    for cat in range(10):
        category_x[cat] = np.array(category_x[cat]).reshape(-1, 28, 28, 1).astype('float32') / 255.
        k = new_max_count - values[cat]
        batch = generate_data(category_x[cat], category_y[cat], k)[0]
        print(f"Category: {cat}  -  {len(batch[0])} images added")
        x_train = np.concatenate([x_train, batch[0]])
        y_train = np.append(y_train, batch[1])

    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train


def visualize_class_measures(train, test, val=[], title="", measure=""):
    barWidth = 0.25
    precision_tab = np.concatenate([np.array(train), np.array(test), np.array(val)])
    y_min = (((min(precision_tab) * 100) // 5) * 5) / 100
    plt.subplots(figsize=(12, 8))
    plt.title(str(title) + " - " + str(measure), fontsize=15)

    labels = [i for i in range(10)]
    br1 = np.arange(len(train))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, train, color='r', width=barWidth, edgecolor='grey', label='trenujący')
    plt.bar(br2, test, color='g', width=barWidth, edgecolor='grey', label='testujący')
    if len(val) > 0:
        plt.bar(br3, val, color='b', width=barWidth, edgecolor='grey', label='walidujący')

    plt.xlabel('Etykieta')
    plt.ylabel('Miara')
    plt.xticks([r + barWidth for r in range(len(train))], labels)
    plt.ylim(y_min, 1.0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{title}/{measure}.jpg')
    # plt.show()
    plt.clf()


def evaluate_model_part(y, y_pred, name="", improvement=False):
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    if not improvement:
        print(name, ': \t', "{:.6f}".format(accuracy), "\t", "{:.6f}".format(precision), "\t", "{:.6f}".format(recall))
        precision_class = precision_score(y, y_pred, average=None)
        recall_class = recall_score(y, y_pred, average=None)
        return precision_class, recall_class
    else:
        return accuracy, precision, recall


def evaluate_model(model_name, name="accuracy"):
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(model_name)
    model = tf.keras.models.load_model(f"improvement/{name}.h5")
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y_val_pred = model.predict(x_val)

    # model evaluation
    print('\t\t\t\t', "accuracy", "\t", "precision", "\t", "recall")
    evaluate_model_part(y_train, y_train_pred, "Trenujący")
    evaluate_model_part(y_test, y_test_pred, "Testujący")
    evaluate_model_part(y_val, y_val_pred, "Walidacyjny")


def visualize_measures(accuracy, precision, recall, title=""):
    barWidth = 0.25
    precision_tab = np.concatenate([np.array(accuracy), np.array(precision), np.array(recall)])
    y_min = (((min(precision_tab) * 100) // 5) * 5) / 100 - 0.02
    plt.subplots(figsize=(12, 8))
    plt.title(str(title), fontsize=15)

    labels = [i for i in range(len(accuracy))]
    br1 = np.arange(len(accuracy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, accuracy, color='r', width=barWidth, edgecolor='grey', label='dokładność')
    plt.bar(br2, precision, color='g', width=barWidth, edgecolor='grey', label='precyzja')
    plt.bar(br3, recall, color='b', width=barWidth, edgecolor='grey', label='pełność')

    plt.xlabel('Epoka')
    plt.ylabel('Miara')
    plt.xticks([r + barWidth for r in range(len(accuracy))], labels)
    plt.ylim(y_min, 1.0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'improvement/{title}.jpg')
    # plt.show()
    plt.clf()
