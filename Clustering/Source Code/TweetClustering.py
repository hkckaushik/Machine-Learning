import re
import sys
import simplejson as json
import random

seeds = []
tweets = []
jaccardDistance = {}
tweetsIds = []
clusters = {}

def loadSeeds(seedsFilePath):
    f = open(seedsFilePath)
    lines = re.split(r',\n',f.read())
    seeds.extend(lines)

def loadDataFile(tweetsDataFile):
    f = open(tweetsDataFile)
    lines = re.split(r'\n', f.read())
    for eachLine in lines:
        tweets.append(json.loads(eachLine))
    newDict = {}
    for eachTweet in tweets:
        newDict[eachTweet['id']] = 1.0
    tweetsIds.extend(newDict.keys())
    #Initialize JaccardDistanceMetric
    for eachID in tweetsIds:
        jaccardDistance[eachID] = dict(newDict)

def calculateJaccardMetric():
    tweetCount = len(tweetsIds)
    for outerIndex in range(tweetCount):
        for innerIndex in range(outerIndex,tweetCount):
            outerID = tweetsIds[outerIndex]
            innerID = tweetsIds[innerIndex]
            tokens = [tweet for tweet in tweets if ((tweet['id'] == outerID) or (tweet['id'] == innerID))]
            if len(tokens) == 1:
                #Same tweet so JaccardMetric = 0
                jaccardDistance[outerID][innerID] = 0
                jaccardDistance[innerID][outerID] = 0
            else:

                #Only two will be there
                tweet1Tokens = set(re.split(r'\W*',tokens[0]['text']))
                tweet2Tokens = set(re.split(r'\W*',tokens[1]['text']))
                unionTokens = tweet1Tokens.union(tweetsDataFile)
                intersectionTokens = tweet1Tokens.intersection(tweet2Tokens)
                jaccardIndex = 1 - (len(intersectionTokens)*1.0/len(unionTokens)*1.0)
                jaccardDistance[outerID][innerID] = jaccardIndex
                jaccardDistance[innerID][outerID] = jaccardIndex

def cluster(numberOfClusters):
    SSE = 0
    for index in range(1,numberOfClusters+1):
        clusters[index] = []
    for eachTweetId in tweetsIds:
        minValue = jaccardDistance[eachTweetId][long(seeds[0])]
        clusterIndex = 1
        for seedIndex in range(1,len(seeds)):
            eachSeed = long(seeds[seedIndex])
            value = jaccardDistance[eachTweetId][eachSeed]
            if value<minValue:
                minValue = value
                clusterIndex = seedIndex+1
        SSE += minValue
        clusters[clusterIndex].append(str(eachTweetId))
    return SSE

def printOutput(outputFileName,clusterCount,SSE):
    with open(outputFileName, 'wb+') as fout:
        for index in range(clusterCount):
            eachCluster = clusters[index+1]
            string = str(index)+" : "
            for eachID in eachCluster:
                string+=str(eachID)+", "
            string = string[:-2]+"\n"
            fout.write(string)
        fout.write("Squared Sum Error (SSE) = "+str(SSE))

if __name__ == "__main__":
    if (len(sys.argv) == 5):
        numberOfClusters = int(sys.argv[1])
        initialSeedsFile = sys.argv[2]
        tweetsDataFile = sys.argv[3]
        outputFileName = sys.argv[4]
        loadSeeds(initialSeedsFile)
        loadDataFile(tweetsDataFile)
        calculateJaccardMetric()
        SSE = cluster(numberOfClusters)
        printOutput(outputFileName,numberOfClusters,SSE)
    else:
        print "Invalid Input arguments"
        print "python.exe TweetClustering.py numberOfClusters initialSeedsFile tweetsDataFile outputFileName"