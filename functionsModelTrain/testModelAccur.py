# from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as all_score

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
