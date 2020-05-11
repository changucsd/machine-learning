#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#




from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


def inputData(file):
    data = np.genfromtxt(file, delimiter=',')
    x = data[:, :3]
    y = data[:, 4]
    return x, y


def plot_error(iterationTimes, errorNumber):
    x = iterationTimes
    y = errorNumber
    plt.plot(x, y)
    plt.xlabel('Iteration Times')
    plt.ylabel('Error number')
    plt.show()
    pass


if __name__ == '__main__':
    iterList = []
    numList = []
    best_score = 0
    W = None
    # X, Y = inputData("classification.txt")

    X = []
    Y = []

    # read file:
    file = open("classification.txt", "r")
    for line in file:
        items = line.split(',')
        X.append([float(items[0]), float(items[1]), float(items[2])])

        if (items[4][0] == '+'):
            Y.append(1)
        else:
            Y.append(0)

    file.close()

    print (X)
    print (Y)

    X = np.c_[np.ones(len(X)), np.array(X)]
    pla = Perceptron()
    pla.n_iter = 1
    pla.warm_start = True
    print(pla.get_params())

    for i in range(0, 7000):
        pla = pla.fit(X, Y)
        score = pla.score(X, Y)
        ErrorNum = (1 - score) * 2000
        iterList.append(i)
        numList.append(ErrorNum)
        if best_score <= score or i == 0:
            best_score = score
            W = pla.coef_

    print('Accuracy =', best_score)
    print('Weights =', W)
    plot_error(iterList, numList)
