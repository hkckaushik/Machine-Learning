import math
import random
import sys
from copy import deepcopy


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.splitOn = None
        self.data = None
        self.isLeafNode = False
        self.Class = None


class DecisionTree:
    def __init__(self):
        pass

    def split(self, data, attribute):
        data0 = []
        data1 = []
        for each in data:
            if (each[attribute] == '1'):
                data1.append(each)
            else:
                data0.append(each)
        return [data0, data1]

    def logbase2(self, x):
        if (x == 0):
            return 0
        else:
            return math.log(x, 2)

    def treeDepth(self, node):
        if (node.isLeafNode is True):
            return 1
        else:
            leftDepth = self.treeDepth(node.left)
            rightDepth = self.treeDepth(node.right)
            maxDepth = max(leftDepth, rightDepth)
            return maxDepth + 1

    def isPureNode(self, data):
        NegativeInstances = 0
        PositiveInstances = 0
        for each in data:
            if (each[classifyingattr] == '0'):
                NegativeInstances += 1
            else:
                PositiveInstances += 1
            if (NegativeInstances != 0 and PositiveInstances != 0):
                return False
        return True

    def getEntropy(self, data):
        positiveInstances = 0.0
        negativeInstances = 0.0
        for each in data:
            if (each[classifyingattr] == '1'):
                positiveInstances += 1
            else:
                negativeInstances += 1
        totalInstances = positiveInstances + negativeInstances
        temp1 = positiveInstances / totalInstances
        temp2 = negativeInstances / totalInstances
        entropy = temp1 * self.logbase2(temp1) + temp2 * self.logbase2(temp2)
        return entropy * -1

    def attributeentropy(self, attribute, data):
        negativeInstances0 = 0.0
        positiveInstances0 = 0.0

        positiveInstances1 = 0.0
        negativeInstances1 = 0.0
        for each in data:
            if (each[attribute] == '1'):
                if (each[classifyingattr] == '1'):
                    positiveInstances1 += 1
                else:
                    negativeInstances1 += 1
            else:
                if (each[classifyingattr] == '1'):
                    positiveInstances0 += 1
                else:
                    negativeInstances0 += 1
        totalInstances1 = positiveInstances1 + negativeInstances1
        totalInstances0 = positiveInstances0 + negativeInstances0

        totalInstances = totalInstances0 + totalInstances1

        if (totalInstances1 != 0):
            temp11 = positiveInstances1 / totalInstances1
            temp12 = negativeInstances1 / totalInstances1
        else:
            temp11 = 1
            temp12 = 1

        if (totalInstances0 != 0):
            temp01 = positiveInstances0 / totalInstances0
            temp02 = negativeInstances0 / totalInstances0
        else:
            temp01 = 1
            temp02 = 1

        entropy1 = temp11 * self.logbase2(temp11) + temp12 * self.logbase2(temp12)
        entropy0 = temp01 * self.logbase2(temp01) + temp02 * self.logbase2(temp02)

        entropy = (totalInstances1 / totalInstances) * entropy1 + (totalInstances0 / totalInstances) * entropy0

        return entropy * -1

    def getSplitAttribute(self, attributes, data):
        splitAttr = None
        minEntropy = 1
        for each in attributes:
            if each != classifyingattr:
                newentropy = self.attributeentropy(each, data)
                if (newentropy < minEntropy):
                    splitAttr = each
                    minEntropy = newentropy
        return splitAttr

    def getNoOfNodes(self, tree, depth=1):
        if tree is None:
            return {'leafNodes': 0, 'totalNodes': 0, 'leafNodeDepthSum': 0}
        if tree.isLeafNode:
            return {'leafNodes': 1, 'totalNodes': 1, 'leafNodeDepthSum': depth}
        leftnodes = self.getNoOfNodes(tree.left, depth + 1)
        rightnodes = self.getNoOfNodes(tree.right, depth + 1)
        return {'leafNodes': (leftnodes['leafNodes'] + rightnodes['leafNodes']),
                'totalNodes': leftnodes['totalNodes'] + rightnodes['totalNodes'] + 1,
                'leafNodeDepthSum': (leftnodes['leafNodeDepthSum'] + rightnodes['leafNodeDepthSum'])}

    def getClass(self, data):
        positiveInstances = 0
        negativeInstances = 0
        for each in data:
            if (each[classifyingattr] == 1):
                positiveInstances += 1
            else:
                negativeInstances += 1
        if positiveInstances > negativeInstances:
            return '1'
        else:
            return '0'

    # Attribute selction using ID3 Algorithm
    def createModel(self, data, attributes):
        node = Node()
        node.data = deepcopy(data)

        if (self.isPureNode(data)):
            node.Class = data[0][classifyingattr]
            node.isLeafNode = True
            return node

        node.Class = self.getClass(data)

        chooseAttributeCount = len(attributes) - 2  # Except Class Attribute
        if (chooseAttributeCount >= 0):
            splitAttr = self.getSplitAttribute(attributes, data)
        else:
            return None

        if splitAttr is None:
            node.isLeafNode = True
            return node

        # Deep copy the attributes as python uses pass by reference in passing variables
        copyAttributes = deepcopy(attributes)
        # Remove the attribute in attributes
        copyAttributes.remove(splitAttr)

        # Split the trainingData into two  - with attributeValue =0 and other  with attributeValue =1
        splitData = self.split(data, splitAttr)
        node.splitOn = splitAttr

        node.left = self.createModel(splitData[0], copyAttributes)
        node.right = self.createModel(splitData[1], copyAttributes)
        if (node.left is None) and (node.right is None):
            node.isLeafNode = True
        return node

    # Random attribute selction
    def createRandomModel(self, data, attributes):
        node = Node()
        node.data = deepcopy(data)

        if (self.isPureNode(data)):
            node.Class = data[0][classifyingattr]
            node.isLeafNode = True
            return node

        node.Class = self.getClass(data)

        chooseAttributeCount = len(attributes) - 2  # Except Class Attribute
        if (chooseAttributeCount >= 0):
            splitAttr = attributes[random.randint(0, chooseAttributeCount)]
        else:
            return None

        if splitAttr is None:
            node.isLeafNode = True
            return node

        # Deep copy the attributes as python uses pass by reference in passing variables
        copyAttributes = deepcopy(attributes)
        # Remove the attribute in attributes
        copyAttributes.remove(splitAttr)

        # Split the trainingData into two  - with attributeValue =0 and other  with attributeValue =1
        splitData = self.split(data, splitAttr)
        node.splitOn = splitAttr

        if (len(splitData[0]) != 0):
            node.left = self.createRandomModel(splitData[0], copyAttributes)
        if (len(splitData[1]) != 0):
            node.right = self.createRandomModel(splitData[1], copyAttributes)
        if (node.left is None) and (node.right is None):
            node.isLeafNode = True
        return node

    def printtree(self, node, depth=0):
        if (node is not None and node.isLeafNode):
            sys.stdout.write(node.Class)
            return None
        i = 0
        str = '\n'
        while (i < depth):
            str += '|\t'
            i += 1
        sys.stdout.flush()
        sys.stdout.write(str + node.splitOn + ' = 0 :')
        self.printtree(node.left, depth + 1)
        sys.stdout.flush()
        sys.stdout.write(str + node.splitOn + ' = 1 :')
        self.printtree(node.right, depth + 1)

    def check(self, data, node):
        instances = 0
        correctPrediction = 0
        wrongPrediction = 0
        for each in data:
            predicted = self.predict(node, each)
            if (predicted == each[classifyingattr]):
                correctPrediction += 1
            else:
                wrongPrediction += 1
            instances += 1

        return ((correctPrediction * 1.0) / instances) * 100

    def predict(self, node, instance):
        if (node.isLeafNode):
            return node.Class
        attribute = node.splitOn
        if (instance[attribute] == '0'):
            if (node.left is not None):
                return self.predict(node.left, instance)
            else:
                return node.Class
        else:
            if (node.right is not None):
                return self.predict(node.right, instance)
            else:
                return node.Class

    def nodeDelete(self, node):
        if (node is not None):
            node.left = None
            node.right = None
            node.isLeafNode = True

    def prune(self, root, pruningFactor, validationdata):

        if root is None:
            return

        nodeDetails = self.getNoOfNodes(root)
        pruneNodes = nodeDetails['totalNodes'] - nodeDetails['leafNodes']
        pruningData = []

        for i in range(1, 50):
            node = deepcopy(root)
            nodestodelete = int(pruneNodes * pruningFactor)
            nodeToDelete = random.sample(range(0, pruneNodes), nodestodelete)
            head = node

            # Create an empty queue for level order traversal
            queue = []

            # Enqueue Root and initialize height
            if (node.left is not None):
                queue.append(node.left)
            if (node.right is not None):
                queue.append(node.right)

            currNode = 0

            while (len(queue) > 0 and nodestodelete > 0):
                # Print front of queue and remove it from queue
                node = queue.pop(0)
                if currNode in nodeToDelete:
                    self.nodeDelete(node)
                    nodestodelete -= 1

                currNode += 1

                # Enqueue left child
                if (node.left is not None) and (node.left.isLeafNode is False):
                    queue.append(node.left)

                # Enqueue right child
                if (node.right is not None) and (node.right.isLeafNode is False):
                    queue.append(node.right)

            pruningData.append(
                {'tree': deepcopy(head), 'accuracy': self.check(validationdata, head), 'nodesRemoved': nodeToDelete})

        maxItem = max(pruningData, key=lambda x: x['accuracy'])
        return maxItem['tree']


if __name__ == "__main__":

    attributes = []
    trainingData = []
    validationdata = []
    testdata = []
    classifyingattr = ""

    arguments = sys.argv
    if (len(arguments) != 5):
        print "Invalid arguments"
        sys.exit()

    trainFile = open(arguments[1], 'r')
    attributes = trainFile.readline().split('\n')[0].split(',')
    attributesLen = len(attributes)
    classifyingattr = attributes[attributesLen - 1]

    for line in trainFile:
        data = line.split('\n')[0].split(',')
        instance = {}
        val = 0
        while val < attributesLen:
            instance[attributes[val]] = data[val]
            val += 1
        trainingData.append(instance)
    trainFile.close()

    validationFile = open(arguments[2], 'r')
    validationFile.readline()

    for line in validationFile:
        data = line.split('\n')[0].split(',')
        instance = {}
        val = 0
        while val < attributesLen:
            instance[attributes[val]] = data[val]
            val += 1
        validationdata.append(instance)
    validationFile.close()

    testFile = open(arguments[3], 'r')
    testFile.readline()

    for line in testFile:
        data = line.split('\n')[0].split(',')
        instance = {}
        val = 0
        while val < attributesLen:
            instance[attributes[val]] = data[val]
            val += 1
        testdata.append(instance)
    testFile.close()

    pruningFactor = float(arguments[4])
    if not (pruningFactor > 0 and pruningFactor < 1):
        print "Invalid pruning Factor input"
        sys.exit()

    tree = DecisionTree()
    node = tree.createModel(trainingData, attributes)
    print "\nDecision Tree Created using ID3 algorithm\n"

    nodeDetails = tree.getNoOfNodes(node)
    avgDepth = (nodeDetails['leafNodeDepthSum']*1.0)/nodeDetails['totalNodes']

    print 'Pre-Pruned Accuracy'
    print '--------------------------------'

    print 'Average Depth using ID3 algorithm: ' + str(avgDepth)
    print 'Number of training instances = ' + str(len(trainingData))
    print 'Number of training attributes = ' + str(len(attributes))

    print 'Total Number of nodes in the tree = ' + str(nodeDetails['totalNodes'])
    print 'Number of leaf nodes in the tree = ' + str(nodeDetails['leafNodes'])
    print 'Accuracy of the model on the training dataset = ' + str(tree.check(trainingData, node)) + '%'
    print ''

    print 'Number of validation instances = ' + str(len(validationdata))
    print 'Number of validation attributes = ' + str(len(attributes))
    print 'Accuracy of the model on the validation dataset before pruning = ' + str(
        tree.check(validationdata, node)) + '%'
    print ''

    print 'Number of testing instances = ' + str(len(testdata))
    print 'Number of testing attributes = ' + str(len(attributes))
    print 'Accuracy of the model on the testing dataset = ' + str(tree.check(testdata, node)) + '%'
    print ''

    tree = DecisionTree()
    node = tree.createRandomModel(trainingData, attributes)
    print "\nDecision Tree Created using Random attribute selection\n"

    nodeDetails = tree.getNoOfNodes(node)
    avgDepth = (nodeDetails['leafNodeDepthSum'] * 1.0) / nodeDetails['leafNodes']

    print 'Pre-Pruned Accuracy'
    print '--------------------------------'

    print 'Average Depth using Random Attribute Selection: ' + str(avgDepth)
    print 'Number of training instances = ' + str(len(trainingData))
    print 'Number of training attributes = ' + str(len(attributes))

    print 'Total Number of nodes in the tree = ' + str(nodeDetails['totalNodes'])
    print 'Number of leaf nodes in the tree = ' + str(nodeDetails['leafNodes'])
    print 'Accuracy of the model on the training dataset = ' + str(tree.check(trainingData, node)) + '%'
    print ''

    print 'Number of validation instances = ' + str(len(validationdata))
    print 'Number of validation attributes = ' + str(len(attributes))
    print 'Accuracy of the model on the validation dataset before pruning = ' + str(
        tree.check(validationdata, node)) + '%'
    print ''

    print 'Number of testing instances = ' + str(len(testdata))
    print 'Number of testing attributes = ' + str(len(attributes))
    print 'Accuracy of the model on the testing dataset = ' + str(tree.check(testdata, node)) + '%'
    print ''

    # tree.printtree(node)

    # node = tree.prune(node,pruningFactor,validationdata)
    #
    # print '\n\nPost-Pruned Accuracy'
    # print '--------------------------------'
    #
    # print 'Number of training instances = ' + str(len(trainingData))
    # print 'Number of training attributes = ' + str(len(attributes))
    # nodeDetails = tree.getNoOfNodes(node)
    # print 'Total Number of nodes in the tree = ' + str(nodeDetails['totalNodes'])
    # print 'Number of leaf nodes in the tree = ' + str(nodeDetails['leafNodes'])
    # print 'Accuracy of the model on the training dataset = ' + str(tree.check(trainingData, node)) + '%'
    # print ''
    #
    # print 'Number of validation instances = ' + str(len(validationdata))
    # print 'Number of validation attributes = ' + str(len(attributes))
    # print 'Accuracy of the model on the validation dataset after pruning = ' + str(
    #     tree.check(validationdata, node)) + '%'
    # print ''
    #
    # print 'Number of testing instances = ' + str(len(testdata))
    # print 'Number of testing attributes = ' + str(len(attributes))
    # print 'Accuracy of the model on the testing dataset = ' + str(tree.check(testdata, node)) + '%'
    # print ''

    # tree.printtree(node)
