import math
import os
import re
import sys

testWordsList = {}
classes = []


def extractVocabulary(trainingPath, maxCategories, stopWordsPath):
    classInfo = {}
    trainingWords = []
    traindir = os.listdir(trainingPath)
    dirIndex = 0
    documentCount = 0
    stopWords = []
    if stopWordsPath is not None:
        f = open(stopWordsPath, 'r')
        regex = re.compile('\\W*')
        stopWords = regex.split(f.read())
        f.close()
    for eachDirName in traindir:
        dirIndex += 1
        dirPath = trainingPath + '//' + eachDirName
        dataFolderPath = os.listdir(dirPath)
        classInfo[eachDirName] = {}
        classInfo[eachDirName]['documentCount'] = documentCount
        classwordList = {}
        classwordCount = 0
        for eachFileName in dataFolderPath:
            filePath = dirPath + '//' + eachFileName
            f = open(filePath, 'r')

            # Remove until Lines is found and remove till that line
            data = f.read().split('Lines')
            if (len(data) > 1):
                data = data[1]
            else:
                data = data[0]
            index = data.index('\n')
            data = data[index:]

            # Split the words and remove the empty words
            tokens = re.split(r'\W*', data)
            tokens = [token.lower() for token in tokens if (len(token) > 1 and token not in stopWords)]
            trainingWords.extend(tokens)
            for eachToken in tokens:
                if (classwordList.has_key(eachToken)):
                    classwordList[eachToken] += 1.0
                    classwordCount += 1.0
                else:
                    classwordList[eachToken] = 2.0
                    classwordCount += 2.0
            documentCount += 1.0
        classInfo[eachDirName]['documentCount'] = documentCount - classInfo[eachDirName]['documentCount']
        classInfo[eachDirName]['wordList'] = classwordList
        classInfo[eachDirName]['wordCount'] = classwordCount
        if (dirIndex == maxCategories):
            break

    trainingWords = list(set(trainingWords))
    return trainingWords, documentCount, classInfo


def TrainNultinomialNB(trainingPath, maxCategories, stopWordsPath=None):
    vocab, totaldocumentCount, classInfo = extractVocabulary(trainingPath, maxCategories, stopWordsPath)
    prior = {}
    condProb = {}
    for eachClass in classInfo:
        prior[eachClass] = classInfo[eachClass]['documentCount'] / totaldocumentCount
        for eachWord in vocab:
            classWordList = classInfo[eachClass]['wordList']
            numerator = 1.0
            if classWordList.has_key(eachWord):
                numerator = classWordList[eachWord]
            denom = classInfo[eachClass]['wordCount']
            if not condProb.has_key(eachWord):
                condProb[eachWord] = {}
            condProb[eachWord][eachClass] = numerator / denom
    return vocab, prior, condProb, classInfo.keys()


def extractTokensFromDoc(vocab, documentPath):
    f = open(documentPath, 'r')

    # Remove until Lines is found and remove till that line
    data = f.read().split('Lines')
    if (len(data) > 1):
        data = data[1]
    else:
        data = data[0]
    index = data.index('\n')
    data = data[index:]

    # Split the words and remove the empty words
    tokens = re.split(r'\W*', data)
    tokens = [token.lower() for token in tokens if (len(token) > 1)]
    tokens = list(set(vocab).intersection(set(tokens)))
    return tokens


def applyMultinomailNB(testingPath, data):
    classes = data['classes']
    prior = data['prior']
    condProb = data['condProb']
    vocab = data['vocab']
    correctPredict = 0
    wrongPredict = 0
    for eachClass in classes:
        classCorrect = 0
        classWrong = 0
        classPath = testingPath + '//' + eachClass
        classdir = os.listdir(classPath)
        for eachFile in classdir:
            wordList = extractTokensFromDoc(vocab, classPath + '//' + eachFile)
            predictedClass = applyMultinomailNBforeachDocument(classes, prior, condProb, wordList)
            if predictedClass == eachClass:
                correctPredict += 1.0
                classCorrect += 1
            else:
                wrongPredict += 1.0
                classWrong += 1
    print 'Accuracy is: ' + str((correctPredict / (correctPredict + wrongPredict)) * 100)


def applyMultinomailNBforeachDocument(classes, prior, condProb, wordList):
    maxScore = -100000000
    score = 0
    predictedClass = ""
    for eachClass in classes:
        score = math.log(prior[eachClass])
        for eachWord in wordList:
            score += math.log(condProb[eachWord][eachClass])
        if score > maxScore:
            maxScore = score
            predictedClass = eachClass
    return predictedClass


if __name__ == "__main__":
    if (len(sys.argv) < 3) and (len(sys.argv) > 4):
        print "Invalid input arguments"
        sys.exit(1)
    trainingPath = sys.argv[1]
    testingPath = sys.argv[2]
    maxCategories = 5
    result = {}
    stopWordsPath = None
    if len(sys.argv) == 4:
        stopWordsPath = sys.argv[3]

    # Train Multinomial NB

    result['vocab'], result['prior'], result['condProb'], result['classes'] = TrainNultinomialNB(trainingPath,
                                                                                                 maxCategories,
                                                                                                 stopWordsPath)

    # Apply Multinomial NB
    applyMultinomailNB(testingPath, result)
