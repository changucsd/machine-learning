#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def readFile(input_file_name):
    dataPoints = []
    labels = []

    file = open(input_file_name, "r")
    for line in file:
        items = line.split(',')
        dataPoints.append([float(items[0]), float(items[1])])

        if items[2][0] == '+':
            labels.append([1])
        else:
            labels.append([-1])

    dataPoints = np.array(dataPoints)
    labels = np.array(labels)

    return dataPoints, labels.flatten()

x, y = readFile("linsep.txt")

clf = SVC(kernel='linear',C=500)
clf.fit(x, y)
print(clf.support_vectors_)

color = ['blue' if c == -1 else 'green' for c in y]

plt.scatter(x[:, 0], x[:, 1], c=color)

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='orange',marker='*')

x_test = np.linspace(0, 0.7)
d = -clf.intercept_/clf.coef_[0][1]
k = -clf.coef_[0][0]/clf.coef_[0][1]
y_test = d + k*x_test
plt.plot(x_test, y_test, 'k')

plt.show()

