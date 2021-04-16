from sklearn.model_selection import train_test_split
from sklearn import datasets
from perceptron import Perceptron
from perceptron import accuracy_score as accuracy

X, y = datasets.make_blobs(n_samples=500,n_features=2,centers=2,cluster_std=1,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

perceptronClassifier = Perceptron(learning_rate=0.01, n_iters=1000)
perceptronClassifier.fit(X_train, y_train)
predictions = perceptronClassifier.predict(X_test)

print("Accuracy:", accuracy(y_test, predictions))