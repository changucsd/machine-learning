#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#



import math
import numpy as np
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def linearRegression(points, Ys):
    points = np.array(points)
    Ys = np.array(Ys)

    # print (points.shape)
    # print (Ys.shape)

    W = np.dot(points.T, points) # 3 * 3
    W = np.linalg.inv(W) # 3*3
    W = np.dot(W, points.T) # 3 * 3000
    W = np.dot(W, Ys)

    return W



dataPoints = []
results = []
# read file:
file = open("linear-regression.txt", "r")
for line in file:
    items = line.split(',')
    dataPoints.append([1, float(items[0]), float(items[1])])

    results.append(float(items[2]))

file.close()

finalweights = linearRegression(dataPoints, results)

print (finalweights)
# print (finalweights[1:])

# predicted = np.dot(dataPoints,finalweights)  # 3000 * 3 + 3 * 1 =  3000 * 1
#
# print (predicted)

# def fun(x, y):
#     return x * finalweights[1] + y * finalweights[2]
#
# fig = plt.figure()
# x = []
# y = []
# z = []
# for zi in dataPoints:
#     print (zi)
#     x.append(zi[1])
#     y.append(zi[2])
# ax = fig.add_subplot(111, projection='3d')
#
# x = np.array(x)
# y = np.array(y)
#
# X,Y= np.meshgrid(x, y)
#
# zs = np.array(fun(np.ravel(X), np.ravel(Y)))
# Z = zs.reshape(X.shape)
#
# ax.plot_surface(X,Y,Z,alpha=0.5)
# ax.scatter(x, y, results, c='blue', marker='o')

# ax.plot_wireframe(x, y, predicted)
# ax.plot_surface(x, y, predicted, alpha=0.3)

# ax.set_zlabel('Z')
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# plt.show()
#
# plt.scatter(dataPoints[:, 1], dataPoints[:, 2], marker='o')
# plt.show()