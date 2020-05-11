#
# Authors:
# Ziqiao Gao 2157371827
# Rui Hu 2350308289
# He Chang 5670527576
# Fanlin Qin 5317973858
#

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import math



def getXandY(dataFrame):
    Y = dataFrame['RainTomorrow']
    X = dataFrame.drop('RainTomorrow', axis=1)

    return X.values, Y.values

def plot_error(iterationTimes, errorNumber):
    x = iterationTimes
    y = errorNumber
    plt.plot(x, y)
    plt.xlabel('Number of trials')
    plt.ylabel('Error cases in Test Dataset')
    plt.show()
    pass


if __name__ == '__main__':
    iterList = []
    numList = []
    best_score = 0
    W = None
    # X, Y = inputData("classification.txt")

    dataSet = pd.read_csv('../data/weatherAUS_APP_NORM.csv', sep=',', header=0)
    # cols = [0, 1, 2]
    # dataSet = dataSet.drop(dataSet.columns[cols], axis=1)

    cols = [1]  # get rid of index and day, but keep month
    dataSet = dataSet.drop(dataSet.columns[cols], axis=1)
    dataSet['Month'] = dataSet['Month'].apply(lambda x: math.floor(x / 4))  # sort month into 1,2,3,4 by season

    msk = np.random.rand(len(dataSet)) < 0.8

    trainData = dataSet[msk]
    testData = dataSet[~msk]

    X_train, Y_train = getXandY(trainData)
    X_test, Y_test = getXandY(testData)

    pla = Perceptron(max_iter=1000, random_state=np.random, warm_start=True)
    print(pla.get_params())

    for i in range(0, 700):
        pla = pla.fit(X_train, Y_train)
        score = pla.score(X_test, Y_test)
        Y_pred = pla.predict(X_test)
        # F1 Measure
        Y_test = pd.Series(Y_test)
        series = Y_test.value_counts()
        null_accuracy = (series[0] / (series[0] + series[1]))
        print('Null Acuuracy: ', str(null_accuracy))
        cm = confusion_matrix(Y_test, Y_pred)
        print(cm)
        print('Confusion matrix\n\n', cm)

        print('\nTrue Positives(TP) = ', cm[0, 0])

        print('\nTrue Negatives(TN) = ', cm[1, 1])

        print('\nFalse Positives(FP) = ', cm[0, 1])

        print('\nFalse Negatives(FN) = ', cm[1, 0])
        TP = cm[0, 0]
        TN = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]
        precision = TP / float(TP + FP)

        print('Precision : {0:0.4f}'.format(precision))
        recall = TP / float(TP + FN)

        print('Recall or Sensitivity : {0:0.4f}'.format(recall))
        f1 = f1_score(Y_test, Y_pred)

        print('f1 : {0:0.4f}'.format(f1))

        ErrorNum = (1 - score) * len(Y_test)
        iterList.append(i)
        numList.append(ErrorNum)
        if best_score <= score or i == 0:
            best_score = score
            W = pla.coef_

    print('Accuracy =', best_score)
    print('Weights =', W)

    plot_error(iterList, numList)