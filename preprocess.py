import numpy as np
import gzip
from struct import unpack


def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    return img


def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab


def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img


def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path, normalize=True, one_hot=True):

    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }

    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }

    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])


x_train_path = './mnist_data/train-images-idx3-ubyte.gz'
y_train_path = './mnist_data/train-labels-idx1-ubyte.gz'
x_test_path = './mnist_data/t10k-images-idx3-ubyte.gz'
y_test_path = './mnist_data/t10k-labels-idx1-ubyte.gz'
(x_train, y_train), (x_test, y_test) = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)
# x_train:[60000, 784], y_train:[60000, 10], x_test:[10000, 784], y_test:[10000, 10]

def batch_generator(x_train, y_train, batch_size=32):
    num_batches = x_train.shape[0] // batch_size
    while True:
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        shuffled_X = x_train[indices]
        shuffled_y = y_train[indices]
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            yield shuffled_X[start_index:end_index], shuffled_y[start_index:end_index]
