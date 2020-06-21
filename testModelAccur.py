from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

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

    target_names = ['0', '1']
    return confusion_matrix(test_labels,predictions,target_names=target_names)
