import numpy as np

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class Perceptron:

    # Init function

    def __init__(self, n_iters=1000, learning_rate=0.01):
        self.numberOfIterations = n_iters
        self.learningRate = learning_rate
        self.weights = None
        self.bias = None
        self.activationFunction = self.activation_function

    # Init function

    # Activation Function

    def activation_function(self, X):
        return np.where(X >= 0, 1, 0)

    # Activation Function

    # Fit method

    def fit(self, X, y):
        # Initializing the values for parameters

        numberOfSamples, numberOfFeatures = X.shape
        self.weights = np.zeros(numberOfFeatures)
        self.bias = 0

        # Initializing the values for parameters

        # Y train modification

        yTrain = np.array([1 if i > 0 else 0 for i in y])

        # Y train modification

        # Prediction for Training

        for n in range(self.numberOfIterations):

            for idx, X_idx in enumerate(X):
                preActivationOutput = np.dot(X_idx, self.weights) + self.bias
                yPredicted = self.activationFunction(preActivationOutput)

                # Updation Calculation

                updationFormula = self.learningRate * (yTrain[idx] - yPredicted)
                self.weights += updationFormula * X_idx
                self.bias += updationFormula
                # Updation Calculation

        # Prediction for Training

    # Fit method

    # Predict method

    def predict(self, X):
        preActivationOutput = np.dot(X, self.weights) + self.bias
        yPredicted = self.activationFunction(preActivationOutput)
        return yPredicted

    # Predict method
