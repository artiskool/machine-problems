from model import Neuron
import numpy as np

class Adaline(Neuron):
  def fit(self, X, y):
    X = self.standardize_features(X)
    X = self.add_bias(X)
    self.init_weights(X)
    lmse = 0
    for epoch in range(self.iterations):
      errors = y - self.activation_function(self.net_input(X))
      self.weights += self.learning_rate * X.T.dot(errors)
      #lmse = (errors ** 2).sum() / 2
      lmse = self.sum_squared_errors(errors)
      #print(errors)
      if lmse <= self.learning_rate:
        break
    return self.weights, epoch

  def sum_squared_errors(self, errors):
    return (errors ** 2).sum() / 2

  def predict(self, X):
    return 1 if self.activation_function(self.net_input(X)) >= 0.0 else 0

  def activation_function(self, X):
    return X

  def standardize_features(self, X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)