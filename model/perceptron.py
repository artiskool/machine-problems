import json
import random
from random import randrange

class Perceptron():
  name = 'perceptron'
  def __init__(self, machine):
    self.weightFilename = './data/weights_{}.json'.format(self.name)
    self.datasetFilename = './data/dataset_{}.json'.format(self.name)
    self.machine = machine
    self.weights = []
    self.loadWeights()
    self.nodes = []
    for key in range(len(machine.selectedCmaps)):
      value = machine.selectedCmaps[key]
      self.nodes.append({'input': value, 'index': key, 'weight': self.weights[key] if key in self.weights else None})

  def loadWeights(self):
    try:
      contents = open(self.weightFilename).read(999999999)
      objects = json.loads(contents)
      if not objects:
        return
    except:
      return
    self.weights = objects
    return self.weights

  def loadTrainingDataset(self):
    try:
      contents = open(self.datasetFilename).read(999999999)
      objects = json.loads(contents)
      if not objects:
        return
    except:
      return
    self.trainingDataset = objects
    return self.trainingDataset

  def classify(self):
    activation = float(self.nodes[0]['weight'])
    for i in range(len(self.nodes)-1):
      activation += float(self.nodes[i+1]['weight']) * int(self.nodes[i+1]['input'])
    return True if activation >= 0 else False # False = consonant, True = vowel

  def train(self):
    self.machine.logSummary('')
    self.machine.logSummary('=== TRAINING DATASET ===')
    dataset = self.loadTrainingDataset()
    splits = 3 # split the dataset by 3 folds, 33% each
    self.learningRate = 0.13
    maxEpoch = 100
    # randomly split the dataset into 3 folds
    folds = []
    datasetCopy = list(dataset)
    foldSize = int(len(dataset) / splits)
    for i in range(splits):
      fold = []
      while len(fold) < foldSize:
        index = randrange(len(datasetCopy)) # randomly pick the index
        fold.append(datasetCopy.pop(index))
      folds.append(fold)
    scores = []
    for fold in folds:
      y = [row[-1] for row in fold] # get all y's
      X = list(folds)
      X.remove(fold)
      X = sum(X, [])
      # train the model
      weights = [random.random() for i in range(len(X[0]))]
      for epoch in range(maxEpoch):
        totalError = self.fit(X, weights)
        if totalError == 0:
          self.machine.logSummary('epoch: {}'.format(epoch))
          self.machine.logSummary('total error: {}'.format(totalError))
          break
      # save the weights for future use
      contents = json.dumps(weights, sort_keys=True, indent=4)
      f = open(self.weightFilename, 'w')
      f.write(contents)
      f.close()
      # try predicting test sets
      testSet = []
      for row in fold:
        rowCopy = list(row)
        rowCopy[-1] = None
        testSet.append(rowCopy)
      predicted = []
      for row in testSet:
        predicted.append(self.stepFunction(row, weights))
      scores.append(self.score(y, predicted))
    self.machine.logSummary('Scores: {}'.format(scores))
    self.machine.logSummary('Average: {}'.format(('%.2f%%' % (sum(scores)/float(len(scores))))))

  def score(self, X, y):
    correct = 0
    for i in range(len(X)):
      if X[i] == y[i]:
        correct += 1
    return correct / float(len(X)) * 100

  def fit(self, X, weights):
    totalError = 0
    for row in X:
      error = row[-1] - self.stepFunction(row, weights) #error = y - activationFunction
      totalError += abs(error)
      weights[0] += self.learningRate * error
      for i in range(len(row)-1):
        weights[i+1] += self.learningRate * error * row[i]
    return totalError

  def stepFunction(self, row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
      activation += weights[i+1] * row[i]
    return 1 if activation >= 0 else 0