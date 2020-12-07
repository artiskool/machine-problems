import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from random import random
from model import Neuron

class Adaline(Neuron):
  def __init__(self, n_iterations=100, random_state=1, learning_rate=0.001):
    self.n_iterations = n_iterations
    self.random_state = random_state
    self.learning_rate = learning_rate

  """
    desired output - sum of error
    computation of total error is not absolute but rather LMSE (least means square error)
    square of sum of error divided by 2
    change total error to LMSE
    for every epoch accummulated LMSE divided input pattern
    compare it with threshold
    threshold = (0.01 or 0.001)
    while (lmse > 0.01 and counter < epoch) continue training
  """
  def fit(self, X, y):
    self.weights = [random() for i in range(len(X[0])+1)] # include bias
    lmse = 0
    for epoch in range(self.n_iterations):
      errors = y - self.activation_function(self.net_input(X))
      weights = self.weights[1:]
      all_errors = self.dot(self.transpose(X), errors)
      for i in range(len(weights)):
        self.weights[i+1] += self.learning_rate * all_errors[i]
      self.weights[0] += self.learning_rate * self.sum_errors(errors)
      lmse = self.sum_errors(errors**2) / 2.0
      if (lmse <= self.learning_rate):
        print(lmse)
        break
    return self.weights, epoch

  def sum_errors(self, errors):
    total = 0
    for error in errors:
      total += error
    return total

  def transpose(self, X):
    return list(map(list, zip(*X)))

  def net_input(self, X):
    weights = self.weights[1:]
    # 8.961276131504245e+58
    output = np.dot(X, weights) + self.weights[0]
    #print('output: ', output)
    return np.dot(X, self.weights[1:]) + self.weights[0]

  # identify function
  def activation_function(self, X):
    return X

  def predict(self, X):
    return 1 if self.activation_function(self.net_input(X)) >= 0.0 else 0