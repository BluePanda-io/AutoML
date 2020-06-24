# from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as all_score
from sklearn import metrics
from sklearn.preprocessing import binarize

import matplotlib.pyplot as plt
import numpy as np

def accuracyScoreModel(modelN,test_data,test_labels):
    test_data = test_data.reshape(-1,1)

    predictions = modelN.predict(test_data)

    accMod = accuracy_score(test_labels,predictions)
    accNoClassifier = 1 - test_labels.mean() # This is the minimum prediction if the system is completly random

    print("Accuracy that the Model has = ",accMod)
    print("Accuracy with no prediction = ",accNoClassifier)
    return accMod


def confussionMatr(modelN,test_data,test_labels):
    test_data = test_data.reshape(-1,1)

    predictions = modelN.predict(test_data)

    return confusion_matrix(test_labels,predictions)


def classification_reportRes(modelN,test_data,test_labels):
    test_data = test_data.reshape(-1,1)

    predictions = modelN.predict(test_data)

    target_names = ['OFF', 'ON']
    return classification_report(test_labels,predictions,target_names=target_names)


def VisualizePredTestResults(modelNN,test_data,test_labels,stepN=15,numberLines = 3):

    test_data = test_data.reshape(-1,1)
    predictions = modelNN.predict(test_data)


    for i in range(len(test_labels)/stepN):
        print(i)
        print("True: ",test_labels[i:i+stepN])
        print("Pred: ",predictions[i:i+stepN])

        ConfMatrix = []
        for kt in range(stepN):
            if (test_labels[i+kt]==1 and predictions[i+kt]==1):
                ConfMatrix.append("TP")
            elif (test_labels[i+kt]==0 and predictions[i+kt]==0):
                ConfMatrix.append("TN")
            elif (test_labels[i+kt]==1 and predictions[i+kt]==0):
                ConfMatrix.append("FN")
            elif (test_labels[i+kt]==0 and predictions[i+kt]==1):
                ConfMatrix.append("FP")

        print("ConM:'array",ConfMatrix)

        if (i>=numberLines):
            break


def binarizationDifferentThreshold(y_pred_prob,threshold):
    y_pred_prob = y_pred_prob.reshape(-1,1)
    y_pred_class = binarize(y_pred_prob,threshold)

    return y_pred_class


def rocCurveDisplay(test_labels,y_pred_prob):
    fpr,tpr,thresholds = metrics.roc_curve(test_labels,y_pred_prob)
    plt.plot(fpr,tpr)
    plt.xlim([0,1.0])
    plt.ylim([0,1.0])
    plt.grid(True)


    x = np.linspace(0, 1, num=20)

    plt.plot(x, x);
    return fpr,tpr,thresholds

def evaluate_threshold(threshold,thresholds,tpr,fpr):
    print("sensitivity",tpr[thresholds>threshold][-1])
    print("Specificity",1-fpr[thresholds>threshold][-1])




















#
