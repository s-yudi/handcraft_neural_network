import numpy as np

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class model:

    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        self.W1 = np.random.normal(size=(input_dimension, hidden_dimension))
        self.b1 = np.random.normal(size=(hidden_dimension,))
        self.W2 = np.random.normal(size=(hidden_dimension, output_dimension))
        self.b2 = np.random.normal(size=(output_dimension,))

    def forward(self, x): # x:[batch_size, input_dimension]
        self.z1 = x@self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1@self.W2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def loss(self, y, y_hat, lambda2=0.00001): # y:[batch_size, output_dimension]
        loss = -np.mean(np.sum(y*np.log(y_hat+1e-7), axis=1), axis=0)
        loss += lambda2*np.sum(self.W1**2)+lambda2*np.sum(self.W2**2)
        return loss

    def step(self, x, y, y_hat, alpha, lambda2): # y, y_hat:[batch_size, output_dimension]
        dscores = y_hat
        dscores[range(x.shape[0]), np.argmax(y, axis=1)] -= 1
        dscores /= x.shape[0]

        dW2 = np.dot(self.a1.T, dscores) + 2*lambda2*self.W2
        db2 = np.sum(dscores, axis=0)
        da1 = np.dot(dscores, self.W2.T)
        dz1 = da1 * sigmoid(self.z1) * (1 - sigmoid(self.z1))

        dW1 = np.dot(x.T, dz1) + 2*lambda2*self.W1
        db1 = np.sum(dz1, axis=0)

        self.W1 -= alpha*dW1
        self.b1 -= alpha*db1
        self.W2 -= alpha*dW2
        self.b2 -= alpha*db2


