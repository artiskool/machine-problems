import math
from random import random, randrange

class Neuron(object):
  def __init__(self, learning_rate = 0.01, iterations = 500):
    self.learning_rate = learning_rate
    self.iterations = iterations

  def fit(self, X, y):
    epoch = 1
    return self.weights, epoch

  def init_weights(self, X, include_bias=True):
    X_len = len(X[0]) + 1 if include_bias else len(X[0])
    self.weights = [random() for i in range(X_len)] # include bias
    return self.weights
  """
  def init_weightss(self, X, include_bias=False):
    random_gen = np.random.RandomState(1)
    X_len = len(X[0]) + 1 if include_bias else len(X[0])
    self.weights = random_gen.normal(loc = 0.0, scale = 0.01, size = X_len)
    return self
  """
  # sum of inputs times its weights
  def net_input(self, X):
    return self.dot(X, self.weights)

  def predict(self, X):
    return self.activation_function(X)

  # step function
  def activation_function(self, X):
    return 1 if self.net_input(X) >= 0.0 else 0

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

  def dot(self, X, Y=None):
    """
    Linear algebra
    Dot product: product of two arrays
    f = np.array([1,2])
    g = np.array([4,5])
    ### (1*4)+(2*5) = 14
    np.dot(f, g)
    """
    X = list(X)
    if type(X[0]) is list:
      totals = []
      for i in range(len(X)):
        total = 0
        for j in range(len(X[i])):
          total += X[i][j] * Y[j]
        totals.append(total)
      return totals
    total = 0
    if Y is not None:
      Y = list(Y)
    for i in range(len(X)):
      total += X[i] * (Y[i] if Y is not None else 1)
    return total

  def transpose(self, X):
    return list(map(list, zip(*X)))