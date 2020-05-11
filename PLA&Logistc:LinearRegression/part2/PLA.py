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
from mpl_toolkits.mplot3d import Axes3D

def inputData(file):
    data = np.genfromtxt(file, delimiter=',')
    x = data[:, :3]
    y = data[:, 3]
    return x, y


def display(x, y, w):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('XLabel')
    ax.set_ylabel('YLabel')
    ax.set_zlabel('ZLabel')

    dataSet = x
    for i in range(len(dataSet)):
        X = dataSet[i][1]
        Y = dataSet[i][2]
        Z = dataSet[i][3]
        if y[i] == 1:
            p = ax.scatter(X, Y, Z, c='c', marker='^')
        else:
            p2 = ax.scatter(X, Y, Z, c='r', marker='o')
    ax.legend([p, p2], ['Label 1 data', 'Label -1 data'])
    xs, ys = np.meshgrid(np.arange(-0.2, 1.2, 0.02), np.arange(-0.2, 1.2, 0.02))
    zs = -(w[0] + w[1] * xs + w[2] * ys) / w[3]
    ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)
    plt.show()


if __name__ == '__main__':
    X, Y = inputData("classification.txt")
    X = np.c_[np.ones(len(X)), np.array(X)]
    pla = Perceptron()
    pla.n_iter = 5000
    print(pla.get_params())
    pla = pla.fit(X, Y)
    accuracy = pla.score(X, Y)
    W = pla.coef_
    print('Accuracy =', accuracy)
    print('Weights =', W)
    display(X, Y, W[0])
