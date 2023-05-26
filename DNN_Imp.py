import numpy as np

class DenseLayer():
    def __init__(self, n_neurons=10, activation_function="ReLU"):
     self.n_neurons = n_neurons
     self.activation_function = activation_function
     self.weights = None
    
    def feed_forward(self, X):
        self.weights = np.random.randn(X.shape[1] + 1, self.n_neurons) * 0.01
        self.input = np.hstack((X, np.ones((X.shape[0], 1))))
        # self.input = np.hstack((np.ones((X.shape[0], 1)), X))

        self.z = np.dot(self.input, self.weights)

        # for future updates
        # if self.activation_function == "ReLU":
        #   self.a = np.maximum(0, self.z)

        self.a = self.z
        return self.a
    
    def back_propagation(self, dA, learning_rate):
      m = self.input.shape[0]
      dZ = dA
      self.dW = np.dot(self.input.T, dZ) / m
      dA_prev = np.dot(dZ, self.weights[:-1].T)
      db = np.sum(dZ, axis=0, keepdims=True) / m
    
      self.weights[:-1] -= learning_rate * self.dW
      self.weights[-1] -= learning_rate * db
    
      return dA_prev


class DenseNetwork():
    def __init__(self, layers):
        self.layers = layers
    
    def add(self, layer):
        self.layers.append(layer)
    
    def fit(self, X, y, learning_rate=0.01, epochs=10):
      for epoch in range(epochs):
        self.output = self.__forward_propagation(X)
        self.__backward_propagation(y, learning_rate)

    def __forward_propagation(self, X):
        temp = X
        for layer in self.layers:
            temp = layer.feed_forward(temp)
        return temp
    
    def __backward_propagation(self, y, learning_rate):
        m = y.shape[0]
        dA = -(y - self.output) / m
        for i in range(len(self.layers)-1, -1, -1):
         dA = self.layers[i].back_propagation(dA, learning_rate) 
    
    def predict(self, X):
        output = self.__forward_propagation(X)
        return output
    
    def loss(self, y):
        m = y.shape[0]
        loss = np.sum(np.square(y - self.output)) / (2 * m)
        return loss
