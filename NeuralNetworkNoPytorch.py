import math
import random
import pickle, gzip

#random.seed(42)

with gzip.open('mnist.pkl.gz') as f:
    ((x_train, y_train), (x_valid,y_valid), _) = pickle.load(f, encoding='latin-1')

sizeOfNetwork = [784, 512, 10]

def initialize_weights(layers):
    weights = []
    for i in range(len(layers) - 1):
        weights.append([[(random.random() - 0.5) * 2 for row in range(layers[i + 1])] for column in range(layers[i])])
    return weights

def initialize_bias(layers):
    bias = []
    for i in range(len(layers) - 1):
        bias.append([(random.random() - 0.5) * 0.1 for row in range(layers[i + 1])])
    return bias

def ReLU(input, isLeaky):
    output = []
    for i in range(len(input)):
        row = []
        for y in range(len(input[0])):
            if input[i][y] < 0:
                if isLeaky:
                    row.append(input[i][y] * 0.1)
                else:
                    row.append(0)
            else:
                row.append(input[i][y])
        output.append(row)
    return output

def ReLU_derivative(input, activations, isLeaky):
    output = []
    for i in range(len(input)):
        row = []
        for y in range(len(input[0])):
            if activations[i][y] > 0:
                row.append(input[i][y])
            else:
                if isLeaky:
                    row.append(input[i][y] * 0.1)
                else:
                    row.append(0)
        output.append(row)
    return output

def softmax(input):
    for i in range(len(input)):
        maxValueRow = max(input[i])
        sumExps = 0
        for y in range(len(input[0])):
            input[i][y] = math.exp(input[i][y] - maxValueRow)
            sumExps += input[i][y]
        for j in range(len(input[0])):
            input[i][j] = input[i][j] / sumExps
    return input

def one_hot_encode(labels):
    output = []
    listUnique = list(sorted(set(labels)))
    for _, label in enumerate(labels):
        pos = 0
        for i in range(len(listUnique)):
            if label == listUnique[i]:
                pos = i
        label = [0] * len(listUnique)
        label[pos] = 1
        output.append(label)
    return output

def cross_entropy(inputs, preds):
    total = 0
    for i in range(len(inputs)):
        for y in range(len(inputs[0])):
            inputs[i][y] = max(inputs[i][y], 1e-15)
            total += preds[i][y] * math.log(inputs[i][y])
    total = -total / len(inputs)
    return total

def binary_cross_entropy(inputs, preds):
    total = 0
    for i in range(len(inputs)):
        # Clipping the input values to avoid log(0)
        clipped_input = max(min(inputs[i], 1 - 1e-15), 1e-15)
        
        total += preds[i] * math.log(clipped_input) + (1 - preds[i]) * math.log(1 - clipped_input)
    total = -total / len(inputs)
    return total

def sigmoid(input):
    output = []
    for i in range(len(input)):
        row = []
        for y in range(len(input[0])):
            row.append(1 / (1 + math.exp(-input[i][y])))
        output.append(row)
    return output

def sigmoid2D(input):
    row = []
    for i in range(len(input)):
        row.append(1 / (1 + math.exp(-input[i])))
    return row

def sigmoid_derivative(difference, lastLayer):
    outputs = []
    for i in range(len(difference)):
        row = []
        for y in range(len(difference[0])):
            row.append((2 * difference[i][y] * lastLayer[i][y] * (1 - lastLayer[i][y])) / len(difference))
        outputs.append(row)
    return outputs

def difference_preds(output, preds):
    outputs = []
    for i in range(len(output)):
        difference = []
        for y in range(len(output[0])):
            difference.append((output[i][y] - preds[i][y]))
        outputs.append(difference)
    return outputs


def differece_preds_binary(output, preds):
    difference = []
    for i in range(len(output)):
        difference.append((output[i] - preds[i]))
    return difference

def transpose_matrix(matrix):
    finalMatrix = []
    for i in range(len(matrix[0])):
        row = []
        for y in range(len(matrix)):
            row.append(matrix[y][i])
        finalMatrix.append(row)
    return finalMatrix

def mse(array):
    total = 0
    for i in range(len(array)):
        for y in range(len(array[0])):
            total += array[i][y] ** 2
    total = total / (len(array) * len(array[0]))
    return total

def mse_derivative(difference):
    outputs = []
    for i in range(len(difference)):
        row = []
        for y in range(len(difference[0])):
            row.append((2 * difference[i][y]) / len(difference))

        outputs.append(row)
    return outputs

def matrix_multiply(x, y, bias):
    result = []
    for i in range(len(x)):
        row = []
        for k in range(len(y[0])):
            sum = 0
            for j in range(len(y)):
                sum += x[i][j] * y[j][k]
            if bias != None:
                row.append(sum + bias[k])
            else:
                row.append(sum)
        result.append(row)
    return result

def bias_backpropagate(outputLayerG):
    bias = []
    for i in range(len(outputLayerG[0])):
        addedBias = 0
        for y in range(len(outputLayerG)):
            addedBias += outputLayerG[y][i]
        bias.append(addedBias)
    return bias

def normalise_difference(differece):
    output = []
    for i in range(len(differece)):
        row = []
        for y in range(len(differece[0])):
            row.append(differece[i][y] / len(differece))
        output.append(row)
    return output

def activation_backpropagate(outputLayerG, weights):
    weightsT = transpose_matrix(weights)
    return matrix_multiply(outputLayerG, weightsT, None)

def weights_backpropagate(outputLayerG, activations):
    activationsT = transpose_matrix(activations)
    return matrix_multiply(activationsT, outputLayerG, None)

def forward_and_backward(inputs, preds, epocs, lr = 0.05, batchsize = 10,  sigmoidFunction = False, mseFunction = False, crossEntropySoftMax = False, binaryCrossEntropy = False, LeakyReLU = False):
    weights = initialize_weights(sizeOfNetwork)
    bias = initialize_bias(sizeOfNetwork)
    saveInputs = inputs
    if crossEntropySoftMax:
            preds = one_hot_encode(preds)
    savePreds = preds
    batchSizeRepeat = 0
    networkLength = len(sizeOfNetwork) - 1
    for e in range(epocs):
        finalLayers = []
        noActivationLayers = []
        noActivationLayers.append(inputs)

        inputs = []
        preds = []
        for i in range(batchsize):
            inputs.append(saveInputs[batchsize])
            preds.append(savePreds[batchsize])
            batchSizeRepeat += 1
            if batchSizeRepeat == len(saveInputs):
                batchSizeRepeat = 0

        for i in range(networkLength):
            if i == 0:
                inputLayer = inputs
            else:
                inputLayer = finalLayers[i - 1]
            
            outcome = matrix_multiply(inputLayer, weights[i], bias[i])
            
            if i != networkLength - 1:
                finalLayers.append(ReLU(outcome, LeakyReLU))
            else:
                if sigmoidFunction or binaryCrossEntropy:
                    finalLayers.append(sigmoid(outcome))
                if crossEntropySoftMax:
                    finalLayers.append(softmax(outcome))
            
            noActivationLayers.append(outcome)

        if not binaryCrossEntropy:
            differece = difference_preds(finalLayers[networkLength - 1], preds)
        else:
            differece = differece_preds_binary(finalLayers[networkLength - 1], preds)

        if mseFunction:
            loss = mse(differece)
        if crossEntropySoftMax:
            loss = cross_entropy(finalLayers[networkLength - 1], preds)
        if binaryCrossEntropy:
            loss = binary_cross_entropy(finalLayers[networkLength - 1], preds)
        
        print(f'Loss is: {loss:.20f}')


        #Backpropagate through network
        biasDerivative = []
        weightsDerivative = []

        if sigmoidFunction:
            lastLayerGrad = sigmoid_derivative(differece, finalLayers[networkLength - 1])
        if crossEntropySoftMax or binaryCrossEntropy:
            lastLayerGrad = normalise_difference(differece)

        for i in range(networkLength, 0, -1):
            biasDerivative.insert(0, bias_backpropagate(lastLayerGrad))
            weightsDerivative.insert(0, weights_backpropagate(lastLayerGrad, noActivationLayers[i - 1]))
            if i != 1:
                lastLayerGrad = ReLU_derivative(activation_backpropagate(lastLayerGrad, weights[i - 1]), noActivationLayers[i - 1], LeakyReLU)

        #Update weights and biases
        for i in range(networkLength):
            for y in range(len(weights[i])):
                for k in range(len(weights[i][0])):
                    weights[i][y][k] -= lr * weightsDerivative[i][y][k]
            for y in range(len(bias[i])):
                bias[i][y] -= lr * biasDerivative[i][y]

forward_and_backward(x_valid, y_valid, epocs=50, crossEntropySoftMax=True, lr=0.1, batchsize=50, LeakyReLU=True)