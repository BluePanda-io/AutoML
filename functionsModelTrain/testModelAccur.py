# from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as all_score
from sklearn import metrics

def accuracyScoreModel(modelN,test_data,test_labels):
    test_data = test_data.reshape(-1,1)

    predictions = modelN.predict(test_data)

    return accuracy_score(test_labels,predictions)


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
