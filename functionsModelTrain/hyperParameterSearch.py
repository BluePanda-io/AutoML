from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier


def hyperParameterSearch(modelN,hyperParameters,gridOrRandom,numbIt,train_data,train_labels):
    if gridOrRandom == "grid":
        mlpc = GridSearchCV(modelN,hyperParameters,return_train_score=False)
    elif gridOrRandom == "randG":
        mlpc = RandomizedSearchCV(modelN,hyperParameters,return_train_score=False,n_iter=numbIt)


    train_data = train_data.reshape(-1,1)


    mlpc.fit(train_data,train_labels)

    return mlpc
