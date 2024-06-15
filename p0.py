import numpy as np

# all this for 1 neuron
inputs = [1, 2, 3, 2.5]  # outputs from previous neuron layer of 3 nodes
weights = [0.2, 0.8, -0.5, 1]  # weights for each input
bias = 2.0  # bias for the neuron

output = (
    inputs[0] * weights[0]
    + inputs[1] * weights[1]
    + inputs[2] * weights[2]
    + inputs[3] * weights[3]
    + bias
)

print(output)

# coding for 3 neurons of a layer, these value are we tune
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2.0
bias2 = 3.0
bias3 = 0.5

outputMultiNeurons = [
    inputs[0] * weights1[0]
    + inputs[1] * weights1[1]
    + inputs[2] * weights1[2]
    + inputs[3] * weights1[3]
    + bias1,
    inputs[0] * weights2[0]
    + inputs[1] * weights2[1]
    + inputs[2] * weights2[2]
    + inputs[3] * weights2[3]
    + bias2,
    inputs[0] * weights3[0]
    + inputs[1] * weights3[1]
    + inputs[2] * weights3[2]
    + inputs[3] * weights3[3]
    + bias3,
]

print(outputMultiNeurons)

weightsAll = [weights1, weights2, weights3]
biasAll = [bias1, bias2, bias3]

# multiple each element of the input with each element of the weights and sum them up
npOutputSingleNeuron = np.dot(inputs, weights) + bias
print(npOutputSingleNeuron)

npOutputMultiNeurons = np.dot(weightsAll, inputs) + biasAll
# 3 by 4 matrix dot 4 by 1 matrix = 3 by 1 matrix
# np.dot(weightsAll, inputs) = [np.dot(weights1, inputs), np.dot(weights2, inputs), np.dot(weights3, inputs)]
print(npOutputMultiNeurons, np.shape(weightsAll), np.shape(np.array(weightsAll).T))

inputsMany = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
layer1Output = np.dot(inputsMany, np.array(weightsAll).T) + biasAll

print(layer1Output)

# another layer of neurons
weightsAll2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13],
]
biasAll2 = [-1, 2, -0.5]

layer2Output = np.dot(layer1Output, np.array(weightsAll2).T) + biasAll2
print(layer2Output)
