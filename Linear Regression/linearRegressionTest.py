import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from linear_regression import meanSquaredError

# Getting a dataset

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Getting a dataset

# Plotting the data

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

# Plotting the data

# Creation of model and prediction of output

regressor = LinearRegression(learning_rate=0.05, n_iters=10000)
regressor.fit(X_train, y_train)
yPrediction = regressor.predict(X_test)

# Creation of model and prediction of output

# Mean Squared Error Checking

mse = meanSquaredError(y_test, yPrediction)
print("MSE:", mse)

# Mean Squared Error Checking

# Plotting the Linear Regression Line Output

regressorLine = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, regressorLine, color='black', linewidth=2, label="Prediction")
plt.show()

# Plotting the Linear Regression Line Output
