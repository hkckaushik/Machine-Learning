import math
import random
import re
import sys

# Training Part
learningRate = 0.1
maxIterations = 500
hiddenLayers = []
dataset = []
attributes = []
attributeValues = {}
hiddenlayerCounts = []
model = []

class Perceptron:
    def __init__(self, isHidden=False, isBias=False, isInput=False, isOutput=False, nodeNumber=0):
        self.isHiddenLayerNode = isHidden
        self.forward = {'node': [], 'weight': []}
        self.backward = {'node': [], 'weight': []}
        self.isBias = isBias
        self.isInput = isInput
        self.isOutput = isOutput
        self.nodeNumber = nodeNumber
        self.value = 0
        self.delta = 0

    def getValue(self, data):
        return self.value

    def calculateSigmoid(self, x):
        try:
            z = math.exp(-x) + 1
            return 1 / z
        except OverflowError:
            print x

class Weight:
    def __init__(self, weight):
        self.weight = weight


class Network:
    def __init__(self):
        self.learningRate = learningRate
        self.model = []
        self.layers = {}
        self.layerCount = 0
        self.inputLength = 0

    def addNodesToModel(self, nodes, nextLayerNodes):
        if len(nodes) == 0:
            self.model = nextLayerNodes
            return
        for eachNode in nodes:
            if (len(eachNode.forward['node']) == 0):
                for nextNode in nextLayerNodes:
                    weight = Weight(random.random())
                    eachNode.forward['node'].append(nextNode)
                    eachNode.forward['weight'].append(weight)
                    nextNode.backward['node'].append(eachNode)
                    nextNode.backward['weight'].append(weight)
            else:
                self.addNodesToModel(eachNode.forward['node'], nextLayerNodes)
                break

    def addBiasNodes(self, nodeList,isInput = False,isOutput = False):
        count = len(nodeList)
        if count == 1:
            bias1 = Perceptron(isHidden=True, isBias=True,isInput=isInput,isOutput=isOutput)
            weight1 = Weight(random.random())
            bias1.forward['node'].append(nodeList[0])
            bias1.forward['weight'].append(weight1)
            nodeList[0].backward['node'].append(bias1)
            nodeList[0].backward['weight'].append(weight1)
            self.layers['layer' + str(len(self.layers) - 1)].insert(0, bias1)
        elif count >= 2:
            bias1 = Perceptron(isHidden=True, isBias=True,isInput=isInput,isOutput=isOutput)
            weight1 = Weight(random.random())
            bias1.forward['node'].append(nodeList[0])
            bias1.forward['weight'].append(weight1)
            nodeList[0].backward['node'].append(bias1)
            nodeList[0].backward['weight'].append(weight1)

            bias2 = Perceptron(isHidden=True, isBias=True,isInput=isInput,isOutput=isOutput)
            weight2 = Weight(random.random())
            bias2.forward['node'].append(nodeList[count - 1])
            bias2.forward['weight'].append(weight2)
            nodeList[count - 1].backward['node'].append(bias2)
            nodeList[count - 1].backward['weight'].append(weight2)
            self.layers['layer' + str(len(self.layers) - 1)].insert(0, bias2)
            self.layers['layer' + str(len(self.layers) - 1)].insert(0, bias1)

    def addLayer(self, nodeCount, createBias, isInput=False, isOutput=False, isHidden=False):
        nodeList = []
        for index in range(0, nodeCount):
            nodeList.append(Perceptron(isInput=isInput, isOutput=isOutput, isHidden=isHidden, nodeNumber=index))
        self.addNodesToModel(self.model, nodeList)
        self.layers['layer' + str(len(self.layers))] = nodeList
        if createBias:
            self.addBiasNodes(nodeList,isInput=isInput,isOutput = isOutput)
        self.layerCount += 1

    def createModel(self, inputCount, hiddenCounts, outputCount):
        self.inputLength = inputCount
        self.addLayer(inputCount, False, isInput=True)
        self.hiddenLayerCount = len(hiddenCounts)
        for index in range(0, self.hiddenLayerCount):
            self.addLayer(hiddenCounts[index], True, isHidden=True)
        self.addLayer(outputCount, True, isOutput=True)

    def runModel(self, instance):
        for index in range(0, self.layerCount):
            currentLayerNodes = self.layers['layer' + str(index)]
            for layerNode in currentLayerNodes:
                if layerNode.isInput:
                    layerNode.value = instance[attributes[layerNode.nodeNumber]]
                elif layerNode.isBias:
                    layerNode.value = 1
                else:
                    backLayerNodeCount = len(layerNode.backward['node'])
                    current = 0
                    nodes = layerNode.backward['node']
                    weights = layerNode.backward['weight']
                    sum = 0
                    while current < backLayerNodeCount:
                        sum += nodes[current].getValue(instance) * weights[current].weight
                        current += 1
                        layerNode.value = layerNode.calculateSigmoid(sum)

    def updateWeights(self, classValues):
        currentLayer = self.layerCount - 1
        error = 0
        while (currentLayer >= 0):
            currentLayerNodes = self.layers['layer' + str(currentLayer)]
            currentNodeCount = len(currentLayerNodes) - 1
            while currentNodeCount >= 0:
                each = currentLayerNodes[currentNodeCount]
                if each.isOutput is True and each.isBias is False:
                    trueValue = int(classValues[each.nodeNumber])
                    each.delta = each.value * (1.0 - each.value) * (trueValue - each.value)*1.0
                elif each.isInput is not True:
                    nextLayerNodeCount = len(each.forward['node'])
                    current = 0
                    nodes = each.forward['node']
                    weights = each.forward['weight']
                    sum = 0
                    while current < nextLayerNodeCount:
                        sum += nodes[current].delta * weights[current].weight
                        weightDelta = learningRate * nodes[current].delta * each.value*1.0
                        weights[current].weight += weightDelta
                        current += 1
                    each.delta = (each.value) * (1.0 - each.value) * sum
                elif each.isInput is True:
                    nextLayerNodeCount = len(each.forward['node'])
                    current = 0
                    nodes = each.forward['node']
                    weights = each.forward['weight']
                    while current < nextLayerNodeCount:
                        weightDelta = learningRate * nodes[current].delta * each.value*1.0
                        weights[current].weight += weightDelta
                        current += 1
                currentNodeCount -= 1
            currentLayer -= 1

    def train(self, data, recordsToTrain, errorTolerance):
        hasNextIteration = True
        iteration = 0
        error = 0
        while (hasNextIteration):
            error = 0
            for index in range(0, recordsToTrain):
                instance = data[index]
                self.runModel(instance)
                result = self.outputValues()
                outputValues = result[0]
                trueValues = data[index]['attribute'+str(self.inputLength+1)]
                for index in range(0, len(outputValues)):
                    diff = int(trueValues[index]) - outputValues[index]
                    error += diff * diff
                self.updateWeights(instance['attribute' + str(self.inputLength + 1)])
            iteration += 1
            error=error/2
            # print error
            hasNextIteration = (not error <= errorTolerance) and (iteration <= maxIterations)
        print 'Training Error :' +str(error)

    def outputValues(self):
        currentLayerNodes = self.layers['layer' + str(self.layerCount - 1)]
        NodeCount = len(currentLayerNodes) - 1
        opValues = []
        current = 0
        while current <= NodeCount:
            each = currentLayerNodes[current]
            if each.isOutput is True and each.isBias is False:
                opValues.append(each.value)
            current += 1
        return opValues, opValues.index(max(opValues))

    def test(self, data, startingRecord):
        correct = 0
        wrong = 0
        error = 0
        length = len(data)
        for index in range(startingRecord, length):
            instance = data[index]
            self.runModel(instance)
            result = self.outputValues()
            outputValues = result[0]
            trueValues = data[index]['attribute' + str(self.inputLength + 1)]
            for index in range(0,len(outputValues)):
                diff = int(trueValues[index]) - outputValues[index]
                error += diff*diff
            maxJ = result[1]
            if int(trueValues[maxJ]) == 1:
                correct += 1
            else:
                wrong += 1
        print 'Test Error : ' + str(error/2)
        print 'Accuracy of the model is: '+str(((correct*1.0)/(correct+wrong))*100) + '%'

    def printNetwork(self):
        hiddenLayer = 0
        array = range(0,self.layerCount)
        for layerIndex in array:
            eachLayer = self.layers['layer' + str(layerIndex)]
            biasNode = 0
            neuron = 0
            if eachLayer[0].isInput:
                print 'Input Layer:'
            elif eachLayer[0].isOutput:
                print 'Output Layer:'
            else:
                hiddenLayer+=1
                print 'Hidden Layer ' + str(hiddenLayer)+':'
            for eachNode in eachLayer:
                weights = eachNode.forward['weight']
                nextNodeCount = len(weights)
                if eachNode.isBias:
                    biasNode +=1
                    print '\tBias Node ' + str(biasNode)+ ' weight:'
                else :
                    neuron+=1
                    print '\tNeuron ' +  str(neuron)+ ' weights:'
                for index in range(0,nextNodeCount):
                    print '\t\t' + str(weights[index].weight)


def createAttributesList(line):
    attributessplit = re.split('[, ]', line)
    attributessplit = [x for x in attributessplit if x != '']
    length = len(attributessplit)
    for i in range(1, length + 1):
        attributes.append('attribute' + str(i))
        attributeValues['attribute' + str(i)] = []
    return attributes


def loadProcessedData(data):
    lines = data.split('\n')
    createAttributesList(lines[0])
    output = ()
    ds = []
    for each in lines:
        attributesValue = re.split(',', each)
        attributesValue = [x for x in attributesValue if x != '']
        length = len(attributesValue)
        if (len(attributes) == length):
            i = 0
            instance = {}
            while i < length - 1:
                instance[attributes[i]] = float(attributesValue[i])
                i += 1
            output = tuple(attributesValue[i])
            instance[attributes[i]] = output
            ds.append(instance)
    while (len(ds)>0):
        length = len(ds)
        index = random.randint(0,length-1)
        dataset.append(ds.pop(index))
    return len(attributes) - 1, len(output)

if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) > 3:
        # Training
        # "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\housingdata\processed.data" 80 0.01 2 2 2
        # "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\irisdata\processed.data" 80 0.01 2 2 2
        # "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\adultdata\processed.data" 80 0.01 2 2 2
        datapath = arguments[1]
        f = open(datapath, 'r')
        inputCount, outputCount = loadProcessedData(f.read())
        trainingDataUse = float(arguments[2])
        recordsToTrain = int(trainingDataUse * len(dataset) / 100)
        errorTolerance = float(arguments[3])
        hiddenLayerCount = int(arguments[4])
        i = 1
        hiddenCounts = []
        while i <= hiddenLayerCount:
            count = int(arguments[4 + i])
            hiddenCounts.append(count)
            i += 1

        network = Network()
        network.createModel(inputCount, hiddenCounts, outputCount)
        network.train(dataset, recordsToTrain, errorTolerance)
        network.test(dataset, recordsToTrain)
        network.printNetwork()
    else:
        print "Invalid Number of Arguments"
