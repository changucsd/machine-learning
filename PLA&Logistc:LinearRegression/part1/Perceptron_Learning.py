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


def randomWeights(dim):
    output = [0]
    for n in range(0, dim-1):
        output.append(random.uniform(-1, 1))
    return output

def accuracyCalulator(weights, dataPoints,Ys):
    weights = np.array(weights)
    passNum = 0
    for i in range(0, len(dataPoints)):
        point = np.array(dataPoints[i])
        value = np.dot(weights.T, point)
        if value >= 0 and Ys[i] == 1:
            passNum = passNum + 1
        elif value < 0 and Ys[i] == -1:
            passNum = passNum + 1
    return (passNum / len(dataPoints))



def perceptron(points, Ys, rate):
    points = np.array(points)
    Ys = np.array(Ys)
    weights = np.array(randomWeights(len(points[0])))
    end = False

    while not end:
        # iteration = iteration + 1;
        # print (iteration)

        violation = False
        for index in range(0,len(points)):
            point = points[index]
            result = Ys[index]
            dotProduct = np.dot(weights.T,point)

            if dotProduct < 0 and result == 1:
                weights = weights + rate * point
                violation = True
                break
            elif dotProduct >= 0 and result == -1:
                weights = weights - rate * point
                violation = True
                break
        if not violation:
            end = True
        else:
            end = False
    return weights


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


dataPoints = []
results = []

# read file:
file = open("classification.txt", "r")
for line in file:
    items = line.split(',')
    dataPoints.append([1, float(items[0]), float(items[1]), float(items[2])])

    if(items[3][0] == '+'):
        results.append(1)
    else:
        results.append(-1)

file.close()

finalweights = perceptron(dataPoints, results, 0.5)



# finalweights = np.array([0,13.690257475466536, -10.958219022570331, -8.212050793588109])
print (finalweights)
print (accuracyCalulator(finalweights, dataPoints,results))

# display(dataPoints, results, finalweights)

#test accuracy
# finalweights = np.array(finalweights)
# print (accuracyCalulator(finalweights, dataPoints,results))

# passNum = 0
# for i in range(0,len(dataPoints)):
#     point = np.array(dataPoints[i])
#     value = np.dot(finalweights.T, point)
#     if value >=0 and results[i] == 1:
#         passNum = passNum + 1
#     elif value < 0 and results[i] == -1:
#         passNum = passNum + 1



# display(dataPoints, results, finalweights)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# x = []
# y = []
# z = []
# for zi in dataPoints:
#     x.append(zi[0])
#     y.append(zi[1])
#     z.append(zi[2])
# ax = plt.subplot(111, projection='3d')
#
# ax.scatter(x, y, z, c='blue')
#
#
# ax.set_zlabel('Z')
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# plt.show()

# plt.scatter(dataPoints[:, 0], dataPoints[:, 1], marker='o')
# plt.show()


