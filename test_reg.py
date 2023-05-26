from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=2, noise=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


X_train, X_test = X_train / 255.0, X_test / 255.0
plt.scatter(X_train[:,1], y_train)
plt.show()

linReg = LinearRegression()
linReg.fit(X_train, y_train)
linReg_pred = linReg.predict(X_train)
#plt.plot(X_train[:1], linReg_pred[:1], color="r")
print("Linear Regression predictions: \n")
print(linReg.predict(X_test))
print(f"\n Linear Regression score: { linReg.score(X_test, y_test )}")

first_hidden = DenseLayer(16)
second_hidden = DenseLayer(8)
output_layer = DenseLayer(1)

dnn = DenseNetwork([first_hidden, second_hidden, output_layer])
dnn.fit(X_train, y_train)
print("Dense Network predictions: \n")
print(dnn.predict(X_test))
dnn.evaluate(X_test, y_test)
