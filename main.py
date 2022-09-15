import numpy as np
import functools

def createWeights(layersizes, initialWeightScale):
    # Make random weights
    weightMatrices = []
    for i in range(len(layersizes) - 1):
        # Inside dimension is the row; has the inputs size. Outside dimension is which row; has the output size.
        newWeights = np.random.normal(scale=initialWeightScale, size=(layersizes[i + 1], layersizes[i]))
        weightMatrices.append(np.array(newWeights))
    return weightMatrices

# Change later to actually give real inputs
def getInput(inputSize):
    inputLayer = np.random.normal(loc = 0.5, scale = 0.5, size=(inputSize, 1))
    return inputLayer

# Change later to actually give real outputs
def getExpectedOutput(inputLayer):
    outputLayer = [[inputLayer[0][0] * 2]]
    return np.array(outputLayer)

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
    # Set up and debug params
    np.random.default_rng()
    batches_cost_print = 100
    
    # Hyper-parameters here:
    layersizes = [1, 5, 5, 1]                                                                   # Number of nodes in each layer; first layer is input, last layer is output.
    initialWeightScale = 0.1                                                                    # Initial weight from -scale to scale on a normal distribution

    leakyReluDropoff = 0.01
    activationFunction = functools.partial(leaky_relu, dropOff = leakyReluDropoff)
    activationFunctionPrime = functools.partial(leaky_relu_prime, dropOff = leakyReluDropoff)
    
    outputActivationFunction = noac
    outputActivationFunctionPrime = noac_prime

    learningRate = 0.1                                                                         # Currently just using a constant learning rate, maybe move to more complicated adams optimizer?
    batchSize = 32
    batches = 800
    coefficientMomentum = 0.5                                                                  # Percentage to be used first time, 1 - this is next time

    # Create weights
    weightMatrices = createWeights(layersizes, initialWeightScale)
 
    totalWeightChanges = [np.zeros(np.shape(weightMatrices[n])) for n in range(len(weightMatrices))]
    for batch in range(batches):
        for wc in totalWeightChanges:
            wc = wc * (1 - coefficientMomentum)
        batchCost = 0
        for datapoint in range(batchSize):
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

            # zprimes is a list of vertical vectors of all zprimes after the first layer, including the last layer, each vertical vector is the layersize
            # aneurons is a list of horizontal vectors of all activated neurons including the first layer but not the last layer, each horizontal vector is the layersize
            # errors is a list of vertical vectors of all neuron errors after the first layer, including the last layer, each vertical vector is the layersize

            # Final layer errors (special case)
            currentErrors = (currentLayer - expectedOutput) * zprimes[-1]
            errors.insert(0, currentErrors)
            #print("Errors: ", currentErrors)
            #print("zprimes: ", zprimes)
            #print("aneurons: ", aneurons)
            
            # Backpropagate errors, go all the way back but NOT to the input nodes
            for i in range(len(layersizes) - 2): # Not including first layer or last layer (already done)
                currentErrors = weightMatrices[-1 - i].T.dot(currentErrors) * zprimes[-2 - i]
                errors.insert(0, currentErrors)

            # Calculate weight changes
            for i in range(len(layersizes) - 1): # Weights are in-between layers
                newWeightChange = errors[i].dot(aneurons[i])
                totalWeightChanges[i] = totalWeightChanges[i] + (newWeightChange * coefficientMomentum)
            #print("Weight changes: ", weightChanges)
            
            # Get cost
            currentCost = 0
            for i in range(layersizes[-1]):
                currentCost += (currentLayer[i][0] - expectedOutput[i][0])**2
            batchCost += currentCost / layersizes[-1]
        batchCost /= batchSize
        if batch % batches_cost_print == 0:
            print("Batch ", batch, " cost: ", batchCost)

        for i in range(len(totalWeightChanges)):
            totalWeightChanges[i] = (totalWeightChanges[i] / batchSize) * learningRate  # I think this is where adam optimizer might come in
            weightMatrices[i] = weightMatrices[i] - totalWeightChanges[i]
    
    userInput = input("Please input a number to multiply: ")
    mnil = np.array([[float(userInput)]])
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

def noac(x):
    return x
def noac_prime(x):
    return 1

# Sigmoid?

if __name__ == "__main__":
    main()
