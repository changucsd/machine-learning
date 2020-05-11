#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#



from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

def inputData(file):
    data = np.genfromtxt(file, delimiter=',')
    x = data[:, :2]
    z = data[:, 2]
    return x, z


def display(x, z, LinRegression):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('XLabel')
    ax.set_ylabel('YLabel')
    ax.set_zlabel('ZLabel')

    X = x[:, 0]
    Y = x[:, 1]
    Z = np.array(z[:])
    ax.scatter(X, Y, Z, c='r', marker='o')

    xs, ys = np.meshgrid(np.arange(X.min() - 0.2, X.max() + 0.2, 0.02), np.arange(Y.min() - 0.2, Y.max() + 0.2, 0.02))
    zs = np.zeros(shape=xs.shape)
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            zs[i][j] = LinRegression.predict([[xs[i][j], ys[i][j]]])
    ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)

    plt.show()


if __name__ == '__main__':
    x, z = inputData("linear-regression.txt")
    L = LinearRegression()
    L.fit(x, z)
    print(str(L.get_params()))
    W = L.coef_
    print("Weights = ", W)
    display(x, z, L)

