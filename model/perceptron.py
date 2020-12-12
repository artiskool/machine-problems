from model import Neuron
import numpy as np

class Perceptron(Neuron):
  def fit(self, X, y):
    X = self.add_bias(X)
    self.init_weights(X)
    for epoch in range(self.iterations):
      total_errors = 0
      for xi, output in zip(X, y):
        update_value = self.learning_rate * (output - self.predict(xi))
        self.weights += update_value * xi
        total_errors += int(update_value != 0.0)
      if total_errors == 0:
        break
    return self.weights, epoch

  # step function
  def activation_function(self, X):
    return 1 if self.net_input(X) >= 0.0 else 0