# todo list:
# Convolutions
# Better input-output functions
# Adam optimizer
# Cupy
# Hyper-parameters from file / cli so I can specify hyper-parameters per task

import numpy as np
import functools
import argparse

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
    inputLayer = np.random.normal(loc = 0, scale = 1, size=(inputSize, 1))
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

def calcInput(currentLayer, weightMatrices, activationFunction, outputActivationFunction):
    for i, weightMatrix in enumerate(weightMatrices):
        currentLayer = calcNextLayer(currentLayer, weightMatrix)
        if i != len(weightMatrices) - 1:
            currentLayer = applyActivationFunction(currentLayer, activationFunction)
    currentLayer = applyActivationFunction(currentLayer, outputActivationFunction)
    return currentLayer

def trainNetwork(batches_cost_print, hidden_layers, hidden_layer_size, activationFunction, activationFunctionPrime, learningRate, batchSize, batches, coefficientMomentum, min_cost_stop):
    # Set up and debug params
    np.random.default_rng()
    batches_cost_print = 100
    
    # More hyper-parameters here (move to cli!):
    layersizes = [hidden_layer_size for i in range(hidden_layers)]                              # Number of nodes in each layer; first layer is input, last layer is output.
    layersizes.insert(0, 1)
    layersizes.append(1)
    initialWeightScale = 0.1                                                                    # Initial weight from -scale to scale on a normal distribution

    outputActivationFunction = noac
    outputActivationFunctionPrime = noac_prime

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
            for i, weightMatrix in enumerate(weightMatrices):
                aneurons.append(currentLayer.copy().T)
                currentLayer = calcNextLayer(currentLayer, weightMatrix)
                if i != len(weightMatrices) - 1:
                    zprimes.append(np.array([[activationFunctionPrime(a[0]) for a in currentLayer]]).T)
                    currentLayer = applyActivationFunction(currentLayer, activationFunction)
            zprimes.append(np.array([[outputActivationFunctionPrime(a[0]) for a in currentLayer]]).T)
            currentLayer = applyActivationFunction(currentLayer, outputActivationFunction)

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
        if batchCost < min_cost_stop:
            print("Minimum cost reached on batch ", batch, " with cost ", batchCost)
            break
        if batch % batches_cost_print == 0:
            print("Batch ", batch, " cost: ", batchCost)

        for i in range(len(totalWeightChanges)):
            totalWeightChanges[i] = (totalWeightChanges[i] / batchSize) * learningRate  # I think this is where adam optimizer might come in
            weightMatrices[i] = weightMatrices[i] - totalWeightChanges[i]
    
    print("Type 'exit' at any time to leave")
    userInput = input("Please input " + str(layersizes[0]) + " number(s) ")
    while userInput != "exit":
        try:
            mnil = np.array([[float(userInput)]])
        except ValueError:
            userInput = input("Please only insert numbers ")
            continue
        userOutput = calcInput(mnil.copy(), weightMatrices, activationFunction, outputActivationFunction)
        print("Expected output: ", getExpectedOutput(mnil))
        print("NN output: ", userOutput)
    
        userInput = input("Please input " + str(layersizes[0]) + " number(s) ")
    
    return weightMatrices

def main():
    # Set up cli and hyper-parameters (not all cli controlled yet)
    parser = argparse.ArgumentParser(description = "Train a new network and test on custom inputs", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--print", type = int, default = 100, help = "number of batches that will run before printing a cost update")
    parser.add_argument("-l", "--layers", type = int, default = 2, help = "number of hidden layers in network")
    parser.add_argument("-s", "--size", type = int, default = 5, help = "number of nodes in each hidden layer")
    parser.add_argument("-a", "--func", default = "leaky", help = "activation function to use, possible values: leaky, relu, none")
    parser.add_argument("-r", "--rate", type = float, default = 0.5, help = "learning rate for network")
    parser.add_argument("-t", "--bsize", type = int, default = 32, help = "size of batches when training network")
    parser.add_argument("-b", "--batches", type = int, default = 3000, help = "maximum batches to train network on")
    parser.add_argument("-m", "--momentum", type = float, default = 0.5, help = "momentum coefficient from 0.5-1, where 1 is no momentum")
    parser.add_argument("-c", "--cost", type = float, default = 10.0**(-25), help = "minimum cost to reach before stopping training")
    args = parser.parse_args()

    np.random.default_rng()
    batches_cost_print = args.print
    
    hidden_layers = args.layers
    hidden_layer_size = args.size

    activationFunction = noac
    activationFunctionPrime = noac_prime

    if args.func == "leaky":
        leakyReluDropoff = 0.01
        activationFunction = functools.partial(leaky_relu, dropOff = leakyReluDropoff)
        activationFunctionPrime = functools.partial(leaky_relu_prime, dropOff = leakyReluDropoff)
    elif args.func == "relu":
        activationFunction = relu
        activationFunction = relu_prime

    learningRate = args.rate                                                                   # Currently just using a constant learning rate, maybe move to more complicated adams optimizer?
    batchSize = args.bsize
    batches = args.batches
    coefficientMomentum = args.momentum                                                        # Percentage to be used first ntime, 1 - this is next time
    min_cost_stop = args.cost

    weightMatrices = trainNetwork(batches_cost_print, hidden_layers, hidden_layer_size, activationFunction, activationFunctionPrime, learningRate, batchSize, batches, coefficientMomentum, min_cost_stop)

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
