
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
# Rui Hu 2350308289


import pandas as pd
import numpy as np
from sklearn import tree, metrics
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def getXandY(dataFrame):
    Y = dataFrame['RainTomorrow']
    X = dataFrame.drop('RainTomorrow', axis=1)

    return X.values, Y.values


dataSet = pd.read_csv('weatherAUS_APP_NORM.csv', sep=',', header=0)
cols = [1]  # get rid of day column
dataSet = dataSet.drop(dataSet.columns[cols], axis=1)
dataSet['Month'] = dataSet['Month'].apply(lambda x: math.floor(x / 4))  # sort month into 1,2,3,4 by season

msk = np.random.rand(len(dataSet)) < 0.8

trainData = dataSet[msk]
testData = dataSet[~msk]

X_train, Y_train = getXandY(trainData)
X_test, Y_test = getXandY(testData)

regressor = tree.DecisionTreeClassifier()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
y_test = pd.Series(Y_test)
print(y_pred)

series = y_test.value_counts()
null_accuracy = (series[0] / (series[0] + series[1]))
print('Null Acuuracy: ', str(null_accuracy))
cm = confusion_matrix(y_test, y_pred)
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
f1 = f1_score(y_test, y_pred)

print('f1 : {0:0.4f}'.format(f1))
print("Accuracy:", regressor.score(X_test, Y_test))
