#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#



from __future__ import print_function
import numpy as np
from sklearn.linear_model import LogisticRegression


def inputData(file):
    data = np.genfromtxt(file, delimiter=',')
    x = data[:, :3]
    y = data[:, 4]
    return x, y


if __name__ == '__main__':
    x, y = inputData("classification.txt")
    x = np.c_[np.ones(len(x)), np.array(x)]
    L = LogisticRegression()
    print(str(L.get_params()))
    L = L.fit(x, y)
    accuracy = L.score(x, y)
    W = L.coef_
    print('Accuracy =', accuracy)
    print('Weights =', W)

