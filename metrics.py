import numpy as np

def acc(y, y_hat): #y:[60000, 10], y_hat:[10000, 10]
    y_label = np.argmax(y, axis=1)
    y_hat_label = np.argmax(y_hat, axis=1)
    return sum(y_label==y_hat_label)/y.shape[0]