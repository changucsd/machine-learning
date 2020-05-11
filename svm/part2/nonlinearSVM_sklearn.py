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

x, y = readFile("nonlinsep.txt")

#'rbf' kernel
clf = SVC(kernel='rbf', C=700)

#'poly' kernel
# clf = SVC(kernel='poly', C=700,degree=2)


clf.fit(x, y)
print(clf.support_vectors_)

color = ['blue' if c == -1 else 'red' for c in y]
plt.scatter(x[:, 0], x[:, 1], c=color)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='orange', marker='*')

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),np.arange(y_min, y_max, 0.2))

z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.contour(xx, yy, z)
plt.show()