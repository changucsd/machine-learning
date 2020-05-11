#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
# Rui Hu 2350308289

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import math
import time
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# //////////// Neural Network/////////////////

def getXandY(dataFrame):
    Y = dataFrame['RainTomorrow']
    X = dataFrame.drop('RainTomorrow', axis=1)

    return X.values, Y.values

N_HIDDEN_SIZE = 100
EPOCHS = 400
LR = 0.1  # learning rate

dataSet = pd.read_csv('weatherAUS_APP_NORM.csv', sep=',', header=0)

cols = [1]  # get rid of day column
dataSet = dataSet.drop(dataSet.columns[cols], axis=1)
dataSet['Month'] = dataSet['Month'].apply(lambda x: math.floor(x / 4))  # sort month into 1,2,3,4 by season
print (dataSet)

#split data into training and testing sets
msk = np.random.rand(len(dataSet)) < 0.8

trainData = dataSet[msk]
testData = dataSet[~msk]

X_train, Y_train = getXandY(trainData)
X_test, Y_test = getXandY(testData)


mlp = MLPClassifier(solver='lbfgs',
                    hidden_layer_sizes=(N_HIDDEN_SIZE,), activation='tanh', learning_rate_init=LR,
                    max_iter=EPOCHS)

start = time.time()
mlp.fit(X_train, Y_train.ravel())

print("Accuracy for train:\n{a}".format(a=mlp.score(X_train, Y_train.ravel())))
print("Accuracy for test:\n{a}".format(a=mlp.score(X_test, Y_test.ravel())))

end = time.time()
print ('Time taken: ', str(end - start))

# Y_pred is the prediction result
# Y_Test is the actual result
Y_pred = mlp.predict(X_test)
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


# dataSet_train = pd.read_csv('weatherAUS_APP_NORM.csv', sep=',', header=0)
# dataSet_test = pd.read_csv('weatherAUS_APP_TEST_NORM.csv', sep=',', header=0)
#
# cols = [1]  # get rid of day
#
# dataSet_train = dataSet_train.drop(dataSet_train.columns[cols], axis=1)
# dataSet_test = dataSet_test.drop(dataSet_test.columns[cols], axis=1)
# # dataSet['Month'] = dataSet['Month'].apply(lambda x: math.floor(x / 4))  # sort month into 1,2,3,4 by season
# print (dataSet_test)
# print (dataSet_train)



# msk = np.random.rand(len(dataSet)) < 0.8
#
# trainData = dataSet[msk]
# testData = dataSet[~msk]

# print (len(trainData))
# print (len(testData))
# print (len(dataSet))

# X_train, Y_train = getXandY(dataSet_train)
# X_test, Y_test = getXandY(dataSet_test)