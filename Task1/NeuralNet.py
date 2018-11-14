import numpy as np

ignoredAttributes = []
keptAttributes = []

def readData():
    data = []
    attributeCount = [0] * 26364
    objectIndex = -1

    #read in training set
    with open('training.txt') as file:
        for line in file:
            values = line.split()
            print(values)

            if int(values[0]) > len(data):
                currObject = [0] * 26364
                data.append(currObject)
                objectIndex += 1

            data[objectIndex][int(values[1]) - 1] = float(values[2])
            attributeCount[int(values[1]) - 1] += 1

    IgnoreAttributes(attributeCount, len(data))
    data = RemapData(data)

    return data

def readLabels():
    data = []
    #read in labels for training set
    with open('label_training.txt') as file:
        for line in file:
            values = line.split()
            label = int(values[0])
            '''
            if(label == -1):
                label = 0
            '''
            data.append([label])

    return data

def IgnoreAttributes(attributeCount, dataCount):
    for index, attribute in enumerate(attributeCount):
        if (float(attribute) / float(dataCount)) < 0.05:
            ignoredAttributes.append(index)
        else:
            keptAttributes.append(index)

def RemapData(data):
    newData = []
    for object in data:
        newObject = []
        for index in keptAttributes:
            newObject.append(object[index])
        newData.append(newObject)
    return newData

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], 1000)
        self.weights2   = np.random.rand(1000, 1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def predict(self, input):
        predictedlayer1 = sigmoid(np.dot(input, self.weights1))
        output = sigmoid(np.dot(predictedlayer1, self.weights2))
        return output


def PrintCSV(data):
    f = open("data.csv", "w")
    for index in range(417):
        f.write(str(index) + ', ')
    f.write('\n')

    for object in data:
        line = ""
        for attributes in object:
            line = line + str(attributes) + ", "
        f.write(line + '\n')
    f.close()


if __name__ == "__main__":
    X = np.array(readData())
    y = np.array(readLabels())
    print(str(y))
    #PrintCSV(X)
    #X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
    #y = np.array([[0],[1],[1],[0]])
    print('Ignored: ' + str(len(ignoredAttributes)))
    print('Kept: ' + str(len(keptAttributes)))

    #'''
    nn = NeuralNetwork(X,y)

    for i in range(500):
        if i % 100 == 0:
            print('Loop: ' + str(i))
        nn.feedforward()
        nn.backprop()

    f = open("output.txt", "w")
    f.write(str(nn.output))
    #'''
