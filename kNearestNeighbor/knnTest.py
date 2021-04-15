from sklearn import datasets
from sklearn.model_selection import train_test_split
from knearestneighbor import kNearestNeighbor
from knearestneighbor import accuracy_score as accuracy

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

k = int(input("Enter the k value\n"))
clf = kNearestNeighbor(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("custom KNN classification accuracy", accuracy(y_test, predictions))