import random
import math
import csv

ignoredAttributes = []
keptAttributes = []

def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def readData(filename, training):
    data = []
    attributeCount = [0] * 26364
    attributeValues = [[] for _ in range(26364)]
    objectIndex = -1

    #read in training set
    with open(filename) as file:
        for line in file:
            values = line.split()
            print(values)

            if int(values[0]) > len(data):
                currObject = [0] * 26364
                data.append(currObject)
                objectIndex += 1

            data[objectIndex][int(values[1]) - 1] = float(values[2])

            if training:
                attributeCount[int(values[1]) - 1] += 1

                if float(values[2]) not in attributeValues[int(values[1]) - 1]:
                    attributeValues[int(values[1]) - 1].append(float(values[2]))

    if training:
        IgnoreAttributes(attributeCount, attributeValues, len(data))

    data = RemapData(data)

    return data

def readLabels(filename, data):
    index = 0
    #read in labels for training set
    with open(filename) as file:
        for line in file:
            values = line.split()
            label = int(values[0])
            data[index].append(label)
            index = index + 1

    return data

def IgnoreAttributes(attributeCount, attributeValues, dataCount):
    for index, attribute in enumerate(attributeCount):
        #print('Attribute Length: ' + str(len(attributeValues[index])))
        if (float(attribute) / float(dataCount)) < 0.05 or len(attributeValues[index]) < 2:
            ignoredAttributes.append(index)
        else:
            keptAttributes.append(index)
    print('Kept: ' + str(len(keptAttributes)))

def RemapData(data):
    newData = []
    for object in data:
        newObject = []
        for index in keptAttributes:
            newObject.append(object[index])
        newData.append(newObject)
    return newData

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    stdev = math.sqrt(variance)
    if stdev == 0.0:
        '''print("stdev: " + str(math.sqrt(variance)))
        print("var: " + str(variance))
        print("avg: " + str(avg))
        print("num: " + str(numbers))'''
        stdev = 0.00000000000001
    return stdev

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def WritePredictions(filename, predictions):
    f = open(filename, "w")
    for value in predictions:
        f.write(str(value) + '\n')

def TestTraining():
    splitRatio = 0.85
    dataset = readData('training.txt', True)
    dataset = readLabels('label_training.txt', dataset)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))

    # prepare model
    summaries = summarizeByClass(trainingSet)

    # test model
    predictions = getPredictions(summaries, testSet)
    print(str(predictions))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))

def EvaluateData():
    dataset = readData('training.txt', True)
    dataset = readLabels('label_training.txt', dataset)
    testset = readData('testing.txt', False)

    # prepare model
    summaries = summarizeByClass(dataset)

    # test model
    predictions = getPredictions(summaries, testset)
    WritePredictions('label_test.txt', predictions)

EvaluateData()
