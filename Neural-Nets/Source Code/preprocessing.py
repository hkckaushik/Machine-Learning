import sys
import re
import math

dataset = []
attributeCategories = {}
attributes = []
attributeValues = {}

def createAttributesList(line):
    attributessplit = re.split('[, ]',line)
    attributessplit = [x for x in attributessplit if x != '']
    length = len(attributessplit)
    for i in range(1,length+1):
        attributes.append('attribute'+str(i))
        attributeCategories['attribute'+str(i)]={}
        attributeValues['attribute' + str(i)] = []
    return attributes

def categorize(attribute,value):
    categoryList = attributeCategories[attribute]
    if not categoryList.has_key(value):
        categoryList[value] = (len(categoryList)) * 1.0
    numericalvalue = categoryList[value]
    return numericalvalue

def loadData(data):
    lines = data.split('\n')
    createAttributesList(lines[0])
    for each in lines:
        attributesValue = re.split('[^a-zA-Z0-9_<>=.-]',each)
        attributesValue = [x for x in attributesValue if x != '']
        length = len(attributesValue)
        if (len(attributes) == length):
            i = 0
            instance = {}
            while i<length:
                numericalvalue = 0
                if re.match(r"[-+]?[\d*.\d*]", attributesValue[i]) is not None:
                    try:
                        numericalvalue = float(attributesValue[i])
                    except ValueError:
                        numericalvalue = categorize(attributes[i], attributesValue[i])
                else:
                    numericalvalue = categorize(attributes[i],attributesValue[i])
                instance[attributes[i]] = numericalvalue
                attributeValues[attributes[i]].append(numericalvalue)
                i += 1
            dataset.append(instance)

def calculateMean(array):
    return sum(array) / len(array)

def calculateSD(array,mean):
    sum = 0.0
    length = len(array)
    for each in array:
        diff = (each-mean)*1.0
        sum += (diff*diff)
    return math.sqrt(sum/length)

def convertToNormalDist(array,attributeNumber):
    hasCategories = (len(attributeCategories[attributes[attributeNumber-1]].keys()) > 0)
    if not hasCategories and attributeNumber != len(attributes)-1:
        i = 0
        length = len(array)
        mean = calculateMean(array)
        stdev = calculateSD(array, mean)
        while(i<length):
            result = (array[i] - mean)/stdev
            array[i] = result
            dataset[i]['attribute'+str(attributeNumber)] = result
            i+=1

def normalize():
    i=0
    for each in attributes:
        array = attributeValues[each]
        convertToNormalDist(array,i)
        i+=1

def standardize():
    classAttribute = attributes[len(attributes) - 1]
    classValues = attributeValues[classAttribute]
    uniqueValues = []
    for each in classValues:
        if not each in uniqueValues:
            uniqueValues.append(each)
    uniqueLength = len(uniqueValues)
    if uniqueLength > 5:
        # Continuous
        classValues.sort()
        length = len(classValues)
        index = 1
        classes = 4
        iter = 1
        arr = []
        while index < (length -1):
            index = int((1.0/classes)*iter*length - 1)
            arr.append(classValues[index])
            iter += 1
        index = 0
        uniqueDict = {}
        while index < uniqueLength:
            iter = 0
            while iter < classes:
                if uniqueValues[index]<=arr[iter]:
                    break
                iter += 1
            str = '0'*(iter)+'1'+'0'*(classes - iter-1)
            # str = '0' * (classes - iter - 1) + '1' + '0' * iter
            uniqueDict[uniqueValues[index]] = str
            index += 1
    else:
        index = 0
        uniqueDict = {}
        while index < uniqueLength:
            str = '0'*index + '1' +'0'*(uniqueLength - index-1)
            uniqueDict[uniqueValues[index]] = str
            index += 1
        index = 0
        length = len(dataset)
    index = 0
    while index < length:
        res = uniqueDict[dataset[index][classAttribute]]
        classValues[index] = res
        dataset[index][classAttribute] = res
        index+=1

def writeOutput(filePath):
    f = open(filePath, 'w')
    for each in dataset:
        string = ""
        for attribute in attributes:
            string += str(each[attribute])+','
        string = string[:-1]+'\n'
        f.write(string)

if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) == 3:
        # Preprocessing
        # "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\irisdata\iris.data" "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\irisdata\processed.data"
        # "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\housingdata\housing.data" "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\housingdata\processed.data"
        # "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\adultdata\adult.data" "C:\Users\Krishna\Desktop\Assignments\2Machine Learning\Assignment 3\adultdata\processed.data"
        datapath = arguments[1]
        f = open(datapath,'r')
        loadData(f.read())
        standardize()
        normalize()
        writeOutput(arguments[2])
    else:
        print "Invalid outputs"