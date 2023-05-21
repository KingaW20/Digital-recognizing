import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def visualize_history(history: tf.keras.callbacks.History, title = "") -> None:
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
    fig.suptitle(title)

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
    plt.savefig(f'{title}/training.jpg')
    plt.show()


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
