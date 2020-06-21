import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

import math

def logisticRegression_classifier(train_data,train_labels):
    logisticRegr = LogisticRegression()
    train_data = train_data.reshape(-1,1)


    newModel = logisticRegr.fit(train_data,train_labels)

    return newModel


def RandomForest_Classifier(train_data,train_labels,num_estimator = 200):
    rfc = RandomForestClassifier(n_estimators=num_estimator)
    train_data = train_data.reshape(-1,1)


    newModel = rfc.fit(train_data,train_labels)

    return newModel


def SVM_classifier(train_data,train_labels,kernelN="NaN",Cn="NaN",gammaN="NaN"):
    if (kernelN=="NaN"):
        clf = svm.SVC()
    else:
        clf = svm.SVC(kernel = kernelN,C = Cn,gamma = gammaN)


    train_data = train_data.reshape(-1,1)


    newModel = clf.fit(train_data,train_labels)

    return newModel


def NeuralNetwork_Classifier(train_data,train_labels,layersUse=[11,11,11],max_iterN=500):
    # layersUse = [11,11,11]
    mlpc = MLPClassifier(hidden_layer_sizes=layersUse,max_iter=max_iterN)
    # mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
    train_data = train_data.reshape(-1,1)


    newModel = mlpc.fit(train_data,train_labels)

    return newModel












#
