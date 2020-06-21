import sklearn
from sklearn.linear_model import LogisticRegression

def logisticRegression_classifier(train_data,train_labels):
    logisticRegr = LogisticRegression()
    train_data = train_data.reshape(-1,1)


    newModel = logisticRegr.fit(train_data,train_labels)

    return newModel
