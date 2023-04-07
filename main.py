import numpy as np
from preprocess import x_train, y_train, x_test, y_test, batch_generator
from neural_network import model
from metrics import acc
import argparse
import pickle
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# x_train:[60000, 784], y_train:[60000, 10], x_test:[10000, 784], y_test:[10000, 10]

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--hiddens', type=int, default=256)
parser.add_argument('--lambda2', type=float, default=0.00001)
args = parser.parse_args()

if args.mode == 'test':
    if os.path.exists(f'./models/model_lr={args.lr}_hiddens={args.hiddens}_lambda2={args.lambda2}.pkl'):
        with open(f'./models/model_lr={args.lr}_hiddens={args.hiddens}_lambda2={args.lambda2}.pkl', 'rb') as f:
            clf = pickle.load(f)
        y_test_hat = clf.forward(x_test)
        test_acc = acc(y_test, y_test_hat)
        print(f'test_acc:{test_acc}')
    else:
        print('model not found!')
    exit()
elif args.mode == 'train':
    np.random.seed(0)
else:
    print('mode not available!')
    exit()

batch_size = 256
samples_generator = batch_generator(x_train, y_train, batch_size=batch_size)
clf = model(input_dimension=x_train.shape[1], hidden_dimension=args.hiddens, output_dimension=y_train.shape[1])

num_epochs = 400
train_loss = []
test_loss = []
test_acc = []
for epoch in range(num_epochs):
    train_epoch_loss = []
    for batch_i in range(x_train.shape[0] // batch_size):
        x_batch, y_batch = next(samples_generator)
        y_hat = clf.forward(x_batch)
        loss = clf.loss(y_batch, y_hat, args.lambda2)
        clf.step(x_batch, y_batch, y_hat, args.lr, args.lambda2)
        train_epoch_loss.append(loss)
    train_loss.append(np.mean(train_epoch_loss))
    y_test_hat = clf.forward(x_test)
    test_loss.append(clf.loss(y_test, y_test_hat, args.lambda2))
    test_acc.append(acc(y_test, y_test_hat))
    print(f'epoch:{epoch+1} train_loss:{train_loss[-1]} test_loss:{test_loss[-1]} test_acc:{test_acc[-1]}')

with open(f'./models/model_lr={args.lr}_hiddens={args.hiddens}_lambda2={args.lambda2}.pkl', 'wb') as f:
    pickle.dump(clf, f)

plt.figure()
plt.plot(np.arange(1, num_epochs + 1), train_loss, label='training_loss')
plt.plot(np.arange(1, num_epochs + 1), test_loss, label='test_loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./plots/Loss_Curve_lr={args.lr}_hiddens={args.hiddens}_lambda2={args.lambda2}.jpg')

plt.figure()
plt.plot(np.arange(1, num_epochs + 1), test_acc, label='test acc')
plt.title('accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('test accuracy')
plt.legend()
plt.savefig(f'./plots/acc_Curve_lr={args.lr}_hiddens={args.hiddens}_lambda2={args.lambda2}.jpg')

plt.figure()
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].hist(np.reshape(clf.W1, (-1, )))
axes[0, 0].set_title('W1')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 1].hist(np.reshape(clf.W2, (-1, )))
axes[0, 1].set_title('W2')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')
axes[1, 0].hist(np.reshape(clf.b1, (-1, )))
axes[1, 0].set_title('b1')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 1].hist(np.reshape(clf.b2, (-1, )))
axes[1, 1].set_title('b2')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')
fig.tight_layout()
plt.savefig(f'./plots/weights_lr={args.lr}_hiddens={args.hiddens}_lambda2={args.lambda2}.jpg')