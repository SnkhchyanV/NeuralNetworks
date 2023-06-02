import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.ones((output_size))
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases

        if self.activation == 'relu':
            return np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-self.z))
        else:
            return self.z

    def backward(self, dz, lr):
        n, m = self.input.shape

        if self.activation == 'relu':
            dz = dz * (self.z > 0)
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-self.z))
            dz = dz * sigmoid * (1 - sigmoid)

        dw = np.dot(self.input.T, dz) / m
        db = np.sum(dz, axis=0) / m
        da = np.dot(dz, self.weights.T) / m

        self.weights -=  lr * dw
        self.biases -=  lr * db

        return da

class DenseNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward_propagation(self, da, lr):
        for layer in reversed(self.layers):
            da = layer.backward(da, lr)

    def __call__(self, X, y, lr, n_epochs, batch_size=32):
        n, m = X.shape
        n_batches = m // batch_size

        print('\n')
        for epoch in range(n_epochs):
            predictions = self.feed_forward(X)
            loss = self.compute_loss(predictions, y)
            print(f'Loss at epoch {epoch} === {loss}')
            da = self.compute_gradient(predictions, y)
            self.backward_propagation(da, lr)

    def compute_loss(self, predictions, y):
        #n, m = y.shape
        #loss = -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) 
        loss = -np.sum(y * np.log(predictions))
        return loss

    def compute_gradient(self, predictions, y):
        n, m = y.shape
        dz = (predictions - y) / m
        return dz

    def predict(self, X):
        predictions = self.feed_forward(X)
        return predictions
