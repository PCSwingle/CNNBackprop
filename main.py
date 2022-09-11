import numpy as np

def createWeights(layersizes, initialWeightScale):
    # Make random weights
    np.random.default_rng()

    weightMatrices = []
    for i in range(len(layersizes) - 1):
        # Inside dimension is the row; has the inputs + 1 for bias node. Outside dimension is which row; has the outputs.
        newWeights = [[np.random.normal(scale=initialWeightScale) for j in range(layersizes[i] + 1)] for k in range(layersizes[i + 1])]
        # Add 0,0,0 ... 1 to preserve bias node as a 1
        zeroes = [0] * layersizes[i]
        zeroes.append(1)
        newWeights.append(zeroes)

        weightMatrices.append(np.array(newWeights))

    return weightMatrices

def getInput(inputSize):
    inputLayer = [np.random.normal(scale=1) for i in range(inputSize)]
    inputLayer.append(1) # Bias node
    return np.array([inputLayer]).T

def calcNextLayer(inputLayer, weightMatrix, activationFunction):
    outputLayer = np.dot(weightMatrix, inputLayer)
    for i in range(outputLayer.shape[0]):
        outputLayer[i][0] = activationFunction(outputLayer[i][0])
    return outputLayer

def main():
    
    # Hyper-parameters here:
    layersizes = [1, 2, 2, 1] # Number of nodes in each layer; first layer is input, last layer is output. All layers will have 1 extra 'bias' node except output.
    initialWeightScale = 0.1    # Initial weight from -scale to scale on a normal distribution
    leakyReluDropoff = 0.01
    activationFunction = lambda x: (x if x > 0 else x * leakyReluDropoff) # Currently activation function will affect bias node; change later

    weightMatrices = createWeights(layersizes, initialWeightScale)

    print("Weights: ", weightMatrices)

    # Get input into first layer
    currentLayer = getInput(layersizes[0])
    
    print("Input: ", currentLayer)

    # Calculate each layer
    for weightMatrix in weightMatrices:
        currentLayer = calcNextLayer(currentLayer, weightMatrix, activationFunction)

    print("Output: ", currentLayer)

if __name__ == "__main__":
    main()
