import random
import math
import numpy as np


def train_test_sets(allBackets, trainRatio, House0, Outliers_train, Outliers_test):

    totalInfoLen = len(allBackets[0]) #1440
    trainDataRange = int(math.floor(totalInfoLen * trainRatio))

    if House0 == 1:
        #seperate data for each house in order to get balanced data for each house
        house0 = allBackets[0]
        house1 = allBackets[1]
        house2 = allBackets[2]

        #shufling data
        random.shuffle(house0)
        random.shuffle(house1)
        random.shuffle(house2)

        #training set
        training_set = house0[:trainDataRange]
        training_set.extend(house1[:trainDataRange])
        training_set.extend(house2[:trainDataRange])
        random.shuffle(training_set)

        #test set
        test_set = house0[trainDataRange:]
        test_set.extend(house1[trainDataRange:])
        test_set.extend(house2[trainDataRange:])
        random.shuffle(test_set)

    else:
        #seperate data for each house in order to get balanced data for each house
        house1 = allBackets[1]
        house2 = allBackets[2]

        #shufling data
        random.shuffle(house1)
        random.shuffle(house2)

        #training set
        training_set = house1[:trainDataRange]
        training_set.extend(house2[:trainDataRange])
        random.shuffle(training_set)

        #test set
        test_set = house1[trainDataRange:]
        test_set.extend(house2[trainDataRange:])
        random.shuffle(test_set)

    if Outliers_train == 0:
        training = np.array(training_set)
        training = [v for v in training if v[1] < 282]
        training = np.array(training)
    else:
        training = np.array(training_set)

    if Outliers_test == 0:
        testing = np.array(test_set)
        testing = [v for v in testing if v[1] < 282]
        testing = np.array(testing)
    else:
        testing = np.array(test_set)



    TRAIN = [0,0]
    TRAIN[0] = training[:,1] #agg data
    TRAIN[1] = training[:,2] #on-off labels

    TEST = [0,0]
    TEST[0] = testing[:,1]
    TEST[1] = testing[:,2]

    #train_data, train_labels, test_data, test_labels
    return TRAIN[0], TRAIN[1], TEST[0], TEST[1]
