import re
import numpy as np

from sklearn import preprocessing,tree
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def load(filename):
    f = open(filename)
    data = f.read()

    buyingle = preprocessing.LabelEncoder()
    maintle = preprocessing.LabelEncoder()
    doorsle = preprocessing.LabelEncoder()
    personsle = preprocessing.LabelEncoder()
    lug_bootle = preprocessing.LabelEncoder()
    safetyle = preprocessing.LabelEncoder()
    classAttrle = preprocessing.LabelEncoder()

    buyingle.fit(['vhigh', 'high', 'med', 'low'])
    maintle.fit(['vhigh', 'high', 'med', 'low'])
    doorsle.fit(['2', '3', '4', '5more'])
    personsle.fit(['2', '4', 'more'])
    lug_bootle.fit(['small', 'med', 'big'])
    safetyle.fit(['low', 'med', 'high'])
    classAttrle.fit(['unacc', 'acc', 'good', 'vgood'])
    enc = [buyingle, maintle, doorsle, personsle, lug_bootle, safetyle, classAttrle]

    preprocessing.LabelEncoder()

    totalAttributesCount = 7
    lines = data.split('\n')
    ds = []
    classValues = []
    for each in lines:
        values = re.split(',', each)
        values = [x for x in values if x != '']
        if (len(values) == totalAttributesCount):
            instance = []
            for i in range(0, totalAttributesCount - 1):
                instance.append(enc[i].transform([values[i]])[0] * 1.0)
            classValues.append(enc[6].transform([values[6]])[0] * 1.0)
        ds.append(instance)
    ds = np.array(ds)
    ds_scaled = preprocessing.scale(ds)
    classValues = np.array(classValues)
    return ds_scaled, classValues

def decisionTree(scaled_data,classValues):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Decision Tree Accuracy: %0.2f" %scores.mean())

def perceptron(scaled_data,classValues):
    clf = Perceptron(n_iter=500, random_state=15, fit_intercept=True, eta0=0.1)
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Perceptron Accuracy: %0.2f" %scores.mean())

def neuralnet(scaled_data,classValues):
    clf = MLPClassifier(solver='sgd', activation='logistic',
                        hidden_layer_sizes = (500),
                        learning_rate_init=0.01,
                        early_stopping=False,max_iter=1000,
                        random_state = 1)
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Neuralnet Accuracy: %0.2f" % scores.mean())

def deeplearning(scaled_data,classValues):
    clf = MLPClassifier(solver='lbfgs',
                        hidden_layer_sizes=(30, 30, 30),
                        early_stopping=False, max_iter=1000,
                        random_state=1, learning_rate='adaptive')
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Deep Learning Accuracy: %0.2f" % scores.mean())

def svm(scaled_data,classValues):
    clf = SVC(kernel='rbf')
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("SVM Accuracy: %0.2f" % scores.mean())

def naivebayes(scaled_data,classValues):
    clf = GaussianNB()
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Naive Bayes Accuracy: %0.2f" % scores.mean())

def logisticregression(scaled_data,classValues):
    clf = LogisticRegression()
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Logistic Regression Accuracy: %0.2f" %scores.mean())

def knn(scaled_data,classValues):
    clf = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("K-Nearest Neighbours Accuracy: %0.2f" %scores.mean())

def bagging(scaled_data,classValues):
    clf = BaggingClassifier(max_samples = 0.5, max_features = 0.5,n_estimators=50)
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Bagging Accuracy: %0.2f" %scores.mean())

def randomforest(scaled_data,classValues):
    clf = RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split = 2, random_state = 0)
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Random Forest Accuracy: %0.2f" %scores.mean())

def adaboost(scaled_data,classValues):
    clf = AdaBoostClassifier(n_estimators=1000)
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("AdaBoost Accuracy: %0.2f" %scores.mean())

def gradientboosting(scaled_data,classValues):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)
    scores = cross_val_score(clf, scaled_data, classValues, cv=10)
    print("Gradient Boosting Accuracy: %0.2f" %scores.mean())

if __name__ == "__main__":
    scaled_data, classValues = load("car.data")

    decisionTree(scaled_data,classValues)
    perceptron(scaled_data,classValues)
    neuralnet(scaled_data,classValues)
    deeplearning(scaled_data,classValues)
    svm(scaled_data,classValues)
    naivebayes(scaled_data,classValues)
    logisticregression(scaled_data,classValues)
    knn(scaled_data,classValues)
    bagging(scaled_data,classValues)
    randomforest(scaled_data,classValues)
    adaboost(scaled_data,classValues)
    gradientboosting(scaled_data,classValues)