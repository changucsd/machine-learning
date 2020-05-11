#
# Authors:
# Ziqiao Gao 2157371827
# Rui Hu 2350308289
# He Chang 5670527576
# Fanlin Qin 5317973858
#

import numpy as np
from sklearn.svm import SVC
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

dataSet = pd.read_csv('../data/weatherAUS_APP.csv', sep=',', header=0)

cols = [1]  # get rid of index and day, but keep month

dataSet = dataSet.drop(dataSet.columns[cols], axis=1)

msk = np.random.rand(len(dataSet)) < 0.8


trainData = dataSet[msk]
testData = dataSet[~msk]


def getXandY(dataFrame):
    Y = dataFrame['RainTomorrow']
    X = dataFrame.drop('RainTomorrow', axis=1)

    return X.values, Y.values


X_train, Y_train = getXandY(trainData)
X_test, Y_test = getXandY(testData)


#'rbf' kernel
# clf = SVC(kernel='rbf', C=700)

#'poly' kernel
clf = SVC(kernel='poly', C=700, degree=2)


clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
#print(precision_recall_fscore_support(Y_test, Y_pred, average="micro"))
print(clf.score(X_test, Y_test))
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

