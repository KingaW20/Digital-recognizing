import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import shuffle

# change values from range (0, 255) to values from range (-1, 1)
def normalize_data(data):
    data = data.astype(np.float32)
    for image_id in range(len(data)):
        data[image_id] = data[image_id] / 255. * 2. - 1.
    return data


def augment_data(x, y, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
    return train_datagen.flow(x, y, batch_size=batch_size)


def train_generator(x, y, batch_size):
    generator = augment_data(x, y, batch_size)
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch])


# to show transformed images
def plot_sample(x_train, y_train, batch_size=1000, n=5):
    count = 0
    for generator in train_generator(x_train, y_train, batch_size):
        x_tra, y_tra = generator
        count += 1
        if count > 0:
            break
    fig = plt.figure(figsize=(10, 10))
    for i in range(x_tra.shape[0]):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(x_tra[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        if (n * n) < (i + 2):
            break
    plt.show()


# LOAD DATA OF TRAINING AND TESTING SET
(train_x, train_y), (test_x, test_y) = mnist.load_data()
# plt.rcParams.update({'font.size': 24})


# STATISTICS - NUMBER OF EACH DIGIT
values = [0] * 10
labels = [0] * 10
for i in range(len(train_y)):
    values[train_y[i]] += 1
print(values)
for i in range(10):
    labels[i] = i

plt.figure(figsize=(9, 6))
plt.title("Liczba poszczególnych cyfr w bazie danych")
plt.barh(labels, values)
for index, value in enumerate(values):
    plt.text(value, index, str(value))
plt.savefig('digit_number.jpg')
# plt.show()


# # CORRELATIONS - image represented category
# images = [0] * 10
# for i in range(10):
#     images[i] = [0] * 28
#     for row in range(28):
#         images[i][row] = [0] * 28
#
# for image in range(len(train_x)):
#     for row in range(28):
#         for col in range(28):
#             images[train_y[image]][row][col] += train_x[image][row][col]
#
# for i in range(10):
#     for row in range(28):
#         for col in range(28):
#             images[i][row][col] //= values[i]
#
# for i in range(10):
#     plt.imshow(images[i], cmap=plt.get_cmap('gray'))
#     plt.title("Digit: {}".format(i))
#     plt.savefig('representations/representation_' + str(i) + '.jpg')
#     # plt.show()
#     # plt.clf()


# NORMALIZATION
data = normalize_data(train_x)


# DATA AUGMENTATION
train_x = train_x.reshape(-1, 28, 28, 1).astype('float32') / 255.
train_y = to_categorical(train_y.astype('float32'))
# plot_sample(train_x[:5], train_y[:5], batch_size=1000, n=5)



# DATA SPLITS
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_size, val_size, test_size = 0.8, 0.1, 0.1

# SPLIT 1 (x_train1, x_val1, x_test1)
x_train1, x_test1, y_train1, y_test1 = \
    train_test_split(train_x, train_y, test_size=test_size, shuffle=True, stratify=train_y)
x_train1, x_val1, y_train1, y_val1 = \
    train_test_split(x_train1, y_train1, test_size=val_size/(train_size+val_size), shuffle=True, stratify=y_train1)

# SPLIT 2 (x_train1, x_val1, x_test1)
(x_train, y_train), (test_x, test_y) = mnist.load_data()
x_train = normalize_data(x_train)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
values = [0] * 10
for i in range(len(x_train)):
    values[y_train[i]] = values[y_train[i]] + 1
print("Categories number: " + str(values))

# split images based on category
category_x = [0] * 10
category_y = [0] * 10
for i in range(10):
    category_x[i] = [0] * values[i]
    category_y[i] = [0] * values[i]
cat_counter = [0] * 10
for i in range(len(x_train)):
    category_x[y_train[i]][cat_counter[y_train[i]]] = x_train[i]
    category_y[y_train[i]][cat_counter[y_train[i]]] = y_train[i]
    cat_counter[y_train[i]] += 1

max_cat_number = max(values)
last_digit = max_cat_number % 10
new_max_count = max_cat_number + 10 - last_digit
for cat in range(10):
    category_x[cat] = np.array(category_x[cat]).reshape(-1, 28, 28, 1).astype('float32') / 255.
    k = new_max_count - values[cat]
    batch = augment_data(category_x[cat], category_y[cat], k)[0]
    print(f"Category: {cat}  -  {len(batch[0])} images added")
    x_train = np.concatenate([x_train, batch[0]])
    y_train = np.append(y_train, batch[1])

x_train, y_train = shuffle(x_train, y_train)

values = [0] * 10
for i in range(len(y_train)):
    values[y_train[i]] = values[y_train[i]] + 1
print("Categories after augmentation: " + str(values))

x_train1, x_test1, y_train1, y_test1 = \
    train_test_split(x_train, y_train, test_size=test_size, shuffle=True, stratify=y_train)
x_train1, x_val1, y_train1, y_val1 = \
    train_test_split(x_train1, y_train1, test_size=val_size/(train_size+val_size), shuffle=True, stratify=y_train1)

# SPLIT 3 (x_train1, x_test1)
x_train1 = np.append(x_train1, x_val1, 0)
y_train1 = np.append(y_train1, y_val1)
exit(0)