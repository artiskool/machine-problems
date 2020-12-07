import math
from random import random, randrange

class Neuron(object):
  def __init__(self, n_iterations=500, learning_rate=0.13):
    self.n_iterations = n_iterations
    self.learning_rate = learning_rate

  def fit(self, X, y):
    self.weights = [random() for i in range(len(X[0])+1)] # include bias
    epoch = 1
    return self.weights, epoch

  # sum of inputs times its weights
  def net_input(self, X):
    weights = self.weights[1:]
    total = 0
    X = list(X)
    for i in range(len(X)):
      total += X[i] * weights[i]
    return total + self.weights[0]

  # step function
  def activation_function(self, X):
    return 1 if self.net_input(X) >= 0.0 else 0

  def predict(self, X):
    return self.activation_function(X)

  def score(self, X, y):
    error_count = 0
    for i in range(len(X)):
      xi = X[i]
      target = y[i]
      output = self.predict(xi)
      if target != output:
        error_count += 1
    data_count = len(X)
    return (data_count - error_count) / data_count

  def train_test_split(self, X, y):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    # 70 train and 30% test
    data_len = len(X)
    train_len = math.ceil(data_len * 0.7)
    test_len = data_len - train_len
    datasetX = list(X)
    datasetY = list(y)
    for _ in range(train_len):
      index = randrange(len(datasetX)) # randomly pick the index
      X_train.append(datasetX.pop(index))
      y_train.append(datasetY.pop(index))
    for _ in range(test_len):
      index = randrange(len(datasetX)) # randomly pick the index
      X_test.append(datasetX.pop(index))
      y_test.append(datasetY.pop(index))
    return X_train, X_test, y_train, y_test