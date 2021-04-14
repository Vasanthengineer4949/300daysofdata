import numpy as np


def accuracy_score(yTrue, yPrediction):
    accuracy = np.sum(yTrue == yPrediction) / len(yTrue)
    return accuracy


class LogisticRegression:

    def __init__(self, n_iters=1000, learning_rate=0.01):
        self.numberOfIterations = n_iters
        self.learningRate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initializing the values for parameters

        numberOfSamples, numberOfFeatures = X.shape
        self.weights = np.zeros(numberOfFeatures)
        self.bias = 0

        # Initializing the values for parameters

        # Gradient Descent code

        for i in range(self.numberOfIterations):
            # Predicted y value

            yHat = np.dot(X, self.weights) + self.bias
            yPredicted = self.sigmoid(yHat)

            # Predicted y value

            # Calculation of Gradients

            dW = 1 / numberOfSamples * np.dot(X.T, (yPredicted - y))
            dB = 1 / numberOfSamples * np.sum(yPredicted - y)

            # Calculation of Gradients

            # Updating the weights and biases

            self.weights -= self.learningRate * dW
            self.bias -= self.learningRate * dB

        # Gradient Descent code

    # Output Prediction

    def predict(self, X):
        yHat = np.dot(X, self.weights) + self.bias
        yPredicted = self.sigmoid(yHat)
        yOutputClass = [1 if i > 0.5 else 0 for i in yPredicted]
        return np.array(yOutputClass)

    # Output Prediction

    # Sigmoid Function

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid  Function
