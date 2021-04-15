import numpy as np
from collections import Counter

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def euclideanDistance(p, q):
    euclideanDistance = np.sqrt(sum(q - p)**2)
    return euclideanDistance

class kNearestNeighbor:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictedOutputs = [self.predictClass(x) for x in X]
        return predictedOutputs

    def predictClass(self, x):
        distances = [euclideanDistance(x, x1) for x1 in self.X_train]
        nearestNeighborIndices = np.argsort(distances)[:self.k]
        nearestNeighborLabels = [self.y_train[i] for i  in nearestNeighborIndices]
        predictedLabel = (Counter(nearestNeighborLabels).most_common())[0][0]
        return predictedLabel



