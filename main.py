import numpy as np

def main():
    
    # Hyper-parameters here:
    layersizes = [1, 10, 10, 1] # Number of nodes in each layer; first layer is input, last layer is output. All layers will have 1 extra 'bias' node except output.
    initialWeightScale = 0.1    # Initial weight from -scale to scale on a normal distribution
    
    # Make random weights
    np.random.default_rng()

    weightMatrices = []
    for i in range(len(layersize - 1)):
        # Inside dimension is the row; has the inputs + 1 for bias node. Outside dimension is which row; has the outputs.
        newWeights = [[np.random.normal(scale=initialWeightScale) for j in range(layersizes[i] + 1)] for k in range(layersizes[i + 1])]
        weightMatrices.append(np.array(newWeights))

    # Get input into first layer

    

if __name__ == "__main__":
    main()
