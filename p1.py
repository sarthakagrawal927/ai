import numpy as np

# 4 features, 3 samples
X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

# values are advised to be between -1 and 1, biases are advised to be 0


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass


# 4 inputs, 5 neurons
layer1 = LayerDense(4, 5)
# layer1 results in 5 outputs, 2 output neurons
layer2 = LayerDense(5, 2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)

# activation function is a function that is applied to the output of a neuron. Eg: ReLU, Sigmoid, Tanh, etc.
