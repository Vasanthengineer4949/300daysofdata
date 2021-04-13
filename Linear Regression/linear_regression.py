import numpy as np

def meanSquaredError(yTrue, yPred):
    return np.mean((yTrue - yPred)**2)

class LinearRegression:

    def __init__(self,n_iters=1000, learning_rate=0.01):
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

            # Predicted y value

            # Calculation of Gradients

            dW = 1 / (numberOfSamples) * np.dot(X.T,(yHat - y))
            dB = 1 / (numberOfSamples) * np.sum(yHat - y)

            # Calculation of Gradients

            # Updating the weights and biases

            self.weights -= self.learningRate * dW
            self.bias -= self.learningRate * dB

        # Gradient Descent code

    def predict(self, X):
        yPredicted = np.dot(X, self.weights) + self.bias
        return yPredicted







