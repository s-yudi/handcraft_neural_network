
# 构建两层神经网络分类器

This repository is the implementation of a handcrafted neural network model for classification of MNIST data.

### Files in the folder

- `mnist_data/`
  - `t10k-images-idx3-ubyte.gz`
  - `t10k-labels-idx1-ubyte.gz`
  - `train-images-idx3-ubyte.gz`
  - `train-labels-idx1-ubyte.gz`
- `models/`
  - `model_lr=0.001_hiddens=256_lambda2=1e-05.pkl`
  - `...`
- `plots/`
  - `acc_Curve_lr=0.001_hiddens=256_lambda2=1e-05.jpg`
  - `Loss_Curve_lr=0.001_hiddens=256_lambda2=1e-05.jpg`
  - `weights_lr=0.001_hiddens=256_lambda2=1e-05.jpg`
- `main.py`
- `metrics.py`
- `neural_network.py`
- `preprocess.py`


### Required packages
The code has been tested running under Python 3.7.0, with the following packages installed (along with their dependencies):
- numpy == 1.15.1
- matplotlib == 2.2.3

### Running Procedure

#### train the model
```
$ python main.py --mode=train --lr=0.001 --hiddens=256 --lambda2=0.00001
```
#### test the model
```
$ python main.py --mode=test --lr=0.001 --hiddens=256 --lambda2=0.00001
```
only if a model with the same hyperparameters are trained, it can be tested

the trained model will be saved in `models/`
and the plots will be saved in `plots/`

