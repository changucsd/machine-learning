#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
# Rui Hu 2350308289

from __future__ import print_function
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import math
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def getXandY(dataFrame):
    Y = dataFrame['RainTomorrow']
    X = dataFrame.drop('RainTomorrow', axis=1)

    # X = X.drop('RISK_MM', axis=1)

    return X.values, Y.values


if __name__ == '__main__':
    dataSet = pd.read_csv('weatherAUS_APP_NORM.csv', sep=',', header=0)

    # print(dataSet.describe())
    cols = [1]  # get rid of index and day, but keep month
    dataSet = dataSet.drop(dataSet.columns[cols], axis=1)
    dataSet['Month'] = dataSet['Month'].apply(lambda x: math.floor(x / 4))  # sort month into 1,2,3,4 by season
    msk = np.random.rand(len(dataSet)) < 0.8
    trainData = dataSet[msk]
    testData = dataSet[~msk]
    # testData = pd.read_csv('weatherAUS_APP_TEST_NORM.csv', sep=',', header=0)
    # testData = testData.drop(dataSet.columns[cols], axis=1)
    # testData['Month'] = testData['Month'].apply(lambda x: math.floor(x / 4))

    X_train, Y_train = getXandY(trainData)
    X_test, Y_test = getXandY(testData)


    # scaler = MinMaxScaler()
    # X_train = pd.DataFrame(X_train)
    # # print(X_train.describe())
    # X_train = scaler.fit_transform(X_train)
    #
    # X_train = pd.DataFrame(X_train)
    # # print(X_train.describe())
    # X_test = scaler.transform(X_test)


    Logistic = LogisticRegression(solver='saga', max_iter=4000)
    print(str(Logistic.get_params()))

    start = time.time()
    Logistic = Logistic.fit(X_train, Y_train)
    ein = Logistic.score(X_train, Y_train)
    accuracy = Logistic.score(X_test, Y_test)
    Y_pred = Logistic.predict(X_test)
    end = time.time()
    print('Logistic EIN =', ein)
    print('Logistic Accuracy =', accuracy)
    print('Time taken: ', str(end - start))

    #accuracy
    #Y_pred is the prediction result
    #Y_Test is the actual result
    Y_pred = Logistic.predict(X_test)
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
    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]
    precision = TP / float(TP + FP)

    print('Precision : {0:0.4f}'.format(precision))
    recall = TP / float(TP + FN)

    print('Recall or Sensitivity : {0:0.4f}'.format(recall))
    f1 = f1_score(Y_test, Y_pred)

    print('f1 : {0:0.4f}'.format(f1))

    #Precision : 0.9733
    # Recall or Sensitivity : 0.9182
    # f1 : 0.6222