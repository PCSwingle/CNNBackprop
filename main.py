import numpy as np
import functools

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

# Change later to actually give real inputs
def getInput(inputSize):
    inputLayer = [np.random.normal(scale = 1) for i in range(inputSize)]
    inputLayer.append(1) # Bias node, should remove from this function soon
    return np.array([inputLayer]).T

# Change later to actually give real outputs
def getExpectedOutput(inputLayer):
    outputLayer = [inputLayer[0][0] * inputLayer[0][0]]
    outputLayer.append(1) # Bias node, should remove from this function soon
    return np.array([outputLayer]).T

def calcNextLayer(inputLayer, weightMatrix):
    outputLayer = np.dot(weightMatrix, inputLayer)
    return outputLayer

def applyActivationFunction(outputLayer, activationFunction):
    for i in range(outputLayer.shape[0]):
        outputLayer[i][0] = activationFunction(outputLayer[i][0])
    return outputLayer

def calcInput(currentLayer, weightMatrices, activationFunction):
    for weightMatrix in weightMatrices: 
        currentLayer = calcNextLayer(currentLayer, weightMatrix)
        currentLayer = applyActivationFunction(currentLayer, activationFunction)
    return currentLayer

def main():
    
    # Hyper-parameters here:
    layersizes = [1, 10, 10, 1]                                                                     # Number of nodes in each layer; first layer is input, last layer is output. All layers will have 1 extra 'bias' node except output.
    initialWeightScale = 0.1                                                                    # Initial weight from -scale to scale on a normal distribution
    leakyReluDropoff = 0.01
    activationFunction = functools.partial(leaky_relu, dropOff = leakyReluDropoff)              # Currently activation function will affect bias node; change later, won't matter for ReLU
    activationFunctionPrime = functools.partial(leaky_relu_prime, dropOff = leakyReluDropoff)
    learningRate = 0.01                                                                         # Currently just using a constant learning rate, maybe move to more complicated adams optimizer?
    batchSize = 10
    batches = 10000
    coefficientMomentum = 0.9                                                                   # Percentage to be used first time, 1 - this is next time

    # Create weights
    weightMatrices = createWeights(layersizes, initialWeightScale)
    
    totalWeightChanges = [np.array([[0 for inner in range(np.shape(weightMatrices[n])[1])] for outer in range(np.shape(weightMatrices[n])[0])]) for n in range(len(weightMatrices))]
    for batchNum in range(batches):
    
        for wc in totalWeightChanges:
            wc = wc * (1 - coefficientMomentum)
        for batch in range(batchSize):

            # Get input into first layer
            currentLayer = getInput(layersizes[0])
            expectedOutput = getExpectedOutput(currentLayer)
            #print("Input: ", currentLayer)
            #print("Expected output: ", expectedOutput)

            # Calculate each layer
            zprimes = []
            aneurons = []
            for weightMatrix in weightMatrices: 
                aneurons.append(currentLayer.copy().T)
                currentLayer = calcNextLayer(currentLayer, weightMatrix)
                zprimes.append(np.array([[activationFunctionPrime(a[0]) for a in currentLayer]]).T)
                currentLayer = applyActivationFunction(currentLayer, activationFunction)

            #print("Output: ", currentLayer)

            # Calculate neuron errors, we want all errors except we don't care about bias neuron or input neurons
            errors = []

            # zprimes is a list of vertical vectors of all zprimes after the first layer, including the last layer, each vertical vector is the layersize + 1 (the bias node)
            # aneurons is a list of horizontal vectors of all activated neurons including the first layer but not the last layer, each horizontal vector is the layersize + 1 (the bias node)
            # errors is a list of vertical vectors of all neuron errors after the first layer, including the last layer, each vertical vector is the layersize + 1 (the bias node error, which is nonsense)

            # Final layer errors (special case)
            currentErrors = (currentLayer - expectedOutput) * zprimes[-1]
            errors.insert(0, currentErrors)
            #print("Errors: ", currentErrors)
            #print("zprimes: ", zprimes)
            #print("aneurons: ", aneurons)
            
            # Backpropagate errors, go all the way back but NOT to the input nodes
            for i in range(len(layersizes) - 2): # Not including first layer or last layer (already done)
                currentErrors = weightMatrices[-1 - i].T.dot(currentErrors) * zprimes[-2 - i] # This will make bias errors; I don't care about them
                errors.insert(0, currentErrors)

            # Calculate weight and bias changes; the last row of weights is all 0's except for the last is 1 to make the bias node persist, so we ignore the bias node erros and always make that row all 0's
            for i in range(len(layersizes) - 1): # Weights are in-between layers
                newWeightChange = errors[i].dot(aneurons[i])
                newWeightChange[-1] = np.array([0 for j in range(np.shape(newWeightChange)[1])])
                totalWeightChanges[i] = totalWeightChanges[i] + (newWeightChange * coefficientMomentum)

            #print("Weight changes: ", weightChanges)

        for i in range(len(totalWeightChanges)):
            totalWeightChanges[i] = (totalWeightChanges[i] / batchSize) * learningRate  # I think this is where adam optimizer might come in
            weightMatrices[i] = weightMatrices[i] - totalWeightChanges[i]
    
    userInput = input("Please input a number to multiply: ")
    mnil = np.array([[float(userInput), 1]]).T
    userOutput = calcInput(mnil.copy(), weightMatrices, activationFunction)
    print("Expected output: ", getExpectedOutput(mnil))
    print("NN output: ", userOutput)

# Possible activation functions and their derivatives
def relu(x):
    return max(0, x)
def relu_prime(x):
    return 1 if x > 0 else 0

def leaky_relu(x, dropOff=0.01):
    return x if x > 0 else x * dropOff
def leaky_relu_prime(x, dropOff=0.01):
    return 1 if x > 0 else dropOff

# Sigmoid?

if __name__ == "__main__":
    main()
