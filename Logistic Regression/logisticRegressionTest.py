import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
from logistic_regression import accuracy_score as accuracy_score

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

logisticRegressor = LogisticRegression(n_iters=1000, learning_rate=0.01)
logisticRegressor.fit(X_train, y_train)
predictions = logisticRegressor.predict(X_test)

print("Logistic Regression Classification Accuracy:", accuracy_score(y_test, predictions))