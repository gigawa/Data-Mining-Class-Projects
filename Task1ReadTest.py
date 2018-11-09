def readData():
    data = []
    objectIndex = -1

    #read in training set
    with open('training.txt') as file:
        for line in file:
            values = line.split()
            print(values)

            if int(values[0]) > len(data):
                currObject = [0] * 27000
                data.append(currObject)
                objectIndex += 1

            data[objectIndex][int(values[1])] = float(values[2])

    #read in labels for training set
    with open('label_training.txt') as file:
        objectID = 0
        for line in file:
            values = line.split()
            data[objectID][-1] = values[0]
            objectID += 1

    return data

def readTestData():
    data = []
    objectIndex = -1

    #read in training set
    with open('test2.txt') as file:
        for line in file:
            values = line.split()
            print(values)

            if int(values[0]) > len(data):
                currObject = [0] * 7
                data.append(currObject)
                objectIndex += 1

            data[objectIndex][int(values[1])] = float(values[2])

    #read in labels for training set
    with open('label_test2.txt') as file:
        objectID = 0
        for line in file:
            values = line.split()
            data[objectID][-1] = values[0]
            objectID += 1

    return data

#data = readData()
#print(data[0][1195])
#print(data[0][-1])
