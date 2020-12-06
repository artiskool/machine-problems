import json
import random
from random import randrange

class ANN():
  def __init__(self, machine):
    self.weightFilename = './data/weights.json'
    self.machine = machine
    self.weights = []
    self.loadWeights()
    self.nodes = []
    for key in range(len(machine.selectedCmaps)):
      value = machine.selectedCmaps[key]
      self.nodes.append({'input': value, 'index': key, 'weight': self.weights[key]})

  def loadWeights(self):
    contents = open(self.weightFilename).read(999999999)
    objects = json.loads(contents)
    if not objects:
      return
    self.weights = objects
    return self.weights

  def loadTrainingDataset(self):
    contents = open('./data/ann_training_dataset.json').read(999999999)
    objects = json.loads(contents)
    if not objects:
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
    splits = 3 # split the dataset into 3
    rate = 0.13
    maxEpoch = 500
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
      trainSet = list(folds)
      trainSet.remove(fold)
      trainSet = sum(trainSet, [])
      testSet = []
      for row in fold:
        rowCopy = list(row)
        rowCopy[-1] = None
        testSet.append(rowCopy)
      predicted = []
      weights = self.trainWeights(trainSet, rate, maxEpoch)
      for row in testSet:
        predicted.append(self.stepFunction(row, weights))
      actual = [row[-1] for row in fold] # get all y's
      # calculate accuracy
      correct = 0
      for i in range(len(actual)):
        if actual[i] == predicted[i]:
          correct += 1
      accuracy = correct / float(len(actual)) * 100
      scores.append(accuracy)
    self.machine.logSummary('Scores: {}'.format(scores))
    self.machine.logSummary('Average: {}'.format(('%.2f%%' % (sum(scores)/float(len(scores))))))

  def trainWeights(self, train, rate, maxEpoch):
    weights = [random.random() for i in range(len(train[0]))]
    for epoch in range(maxEpoch):
      totalError = 0
      for row in train:
        error = row[-1] - self.stepFunction(row, weights) #error = y - stepFunction
        totalError += abs(error)
        weights[0] += rate * error
        for i in range(len(row)-1):
          weights[i+1] += rate * error * row[i]
      if totalError == 0:
        self.machine.logSummary('epoch: {}'.format(epoch))
        self.machine.logSummary('total error: {}'.format(totalError))
        break
    contents = json.dumps(weights, sort_keys=True, indent=4)
    f = open(self.weightFilename, 'w')
    f.write(contents)
    f.close()
    return weights

  def stepFunction(self, row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
      activation += weights[i+1] * row[i]
    return 1 if activation >= 0 else 0