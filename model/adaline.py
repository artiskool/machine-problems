import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from model import Neuron

class CustomAdaline(Neuron):
  def __init__(self, n_iterations=100, random_state=1, learning_rate=0.001):
    self.n_iterations = n_iterations
    self.random_state = random_state
    self.learning_rate = learning_rate

  '''
  Batch Gradient Descent 
  1. Weights are updated considering all training examples.
  2. Learning of weights can continue for multiple iterations
  3. Learning rate needs to be defined
  '''
  def fit_original(self, X, y):
    rgen = np.random.RandomState(self.random_state)
    self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    for _ in range(self.n_iterations):
      activation_function_output = self.activation_function(self.net_input(X))
      errors = y - activation_function_output
      self.weights[1:] = self.weights[1:] + self.learning_rate * X.T.dot(errors)
      self.weights[0] = self.weights[0] + self.learning_rate*errors.sum()

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
    rgen = np.random.RandomState(self.random_state)
    size = len(X[0]) + 1
    self.weights = rgen.normal(loc=0.0, scale=0.01, size=size)
    for _ in range(self.n_iterations):
      activation_function_output = self.activation_function(self.net_input(X))
      errors = y - activation_function_output
      weights = self.weights[1:]
      all_errors = self.dot(self.transpose(X), errors)
      for i in range(len(weights)):
        self.weights[i+1] = self.weights[i+1] + self.learning_rate * all_errors[i]
      self.weights[0] += self.learning_rate * self.sum_errors(errors)

  def sum_errors(self, errors):
    total = 0
    for error in errors:
      total += error
    return total

  def transpose(self, X):
    return list(map(list, zip(*X)))

  '''
  Net Input is sum of weighted input signals
  '''
  def net_input(self, X):
    weights = self.weights[1:]
    # 8.961276131504245e+58
    output = np.dot(X, weights) + self.weights[0]
    #print('output: ', output)
    return np.dot(X, self.weights[1:]) + self.weights[0]

  '''
  Activation function is fed the net input. As the activation function is
  an identity function, the output from activation function is same as the
  input to the function.
  '''
  def activation_function(self, X):
    return X

  '''
  Prediction is made on the basis of output of activation function
  '''
  def predict(self, X):
    return 1 if self.activation_function(self.net_input(X)) >= 0.0 else 0
    #return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0)


# Load the data set
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target
#X = list(bc.data)
#y = list(bc.target)
# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Instantiate CustomPerceptron
adaline = CustomAdaline(n_iterations = 10)
# Fit the model
adaline.fit(X_train, y_train)

# Score the model
print(adaline.score(X_test, y_test)) # 0.6257309941520468
print(adaline.score(X_train, y_train)) # 0.628140703517588
print('weights: ', adaline.weights)
"""
weights:  [1.19655006e+53 1.92479219e+54 2.42270093e+54 1.26152806e+55
 1.01521468e+56 1.16481282e+52 1.43129731e+52 1.44120458e+52
 7.99759000e+51 2.19191529e+52 7.37904912e+51 5.96311233e+52
 1.38839669e+53 4.23180141e+53 6.74529410e+54 7.82883378e+50
 3.28404315e+51 4.25076593e+51 1.54113033e+51 2.39880806e+51
 4.44207251e+50 2.28607530e+54 3.24181077e+54 1.51799674e+55
 1.45140186e+56 1.61310055e+52 3.59011352e+52 4.18514076e+52
 1.72960940e+52 3.60352931e+52 1.01719338e+52]
 """