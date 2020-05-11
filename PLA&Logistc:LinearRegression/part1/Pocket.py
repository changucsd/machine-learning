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
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

def randomWeights(dim):
    output = [0]
    for n in range(0, dim-1):
        output.append(random.uniform(-1, 1))
    return output


def missNumber(weights, dataPoints, Ys):
    weights = np.array(weights)
    missNum = 0
    for i in range(0, len(dataPoints)):
        point = np.array(dataPoints[i])
        value = np.dot(weights.T, point)
        if value >= 0 and Ys[i] == -1:
            missNum = missNum + 1
        elif value < 0 and Ys[i] == 1:
            missNum = missNum + 1
    return missNum


def pocket(points, Ys, rate, runtime):
    points = np.array(points)
    Ys = np.array(Ys)
    weights = np.array(randomWeights(len(points[0])))
    end = False
    original = runtime

    misclassifiedList = []
    iterations = []

    bestWeights = np.array([])
    lowestMissNum = len(dataPoints)

    while not end and runtime > 0:
        # could be improvement
        # if runtime < original/2:
        #     rate = rate/2

        violation = False

        # caluculate the misscount for current weights
        missCount = missNumber(weights, dataPoints, results)
        # record miss count and iteration index for printing
        misclassifiedList.append(missCount)
        iterations.append(original - runtime)

        if missCount < lowestMissNum:
            lowestMissNum = missCount
            bestWeights = weights

        for index in range(0, len(points)):
            point = points[index]
            result = Ys[index]
            dotProduct = np.dot(weights.T, point)
            if dotProduct < 0 and result == 1:
                weights = weights + rate * point
                violation = True
                break
            elif dotProduct >= 0 and result == -1:
                weights = weights - rate * point
                violation = True
                break
        # if there is no violation, end the while loop
        if not violation:
            end = True
        else:
            end = False
        runtime = runtime - 1

    return bestWeights, lowestMissNum, original - runtime, misclassifiedList, iterations


dataPoints = []
results = []

# read file:
file = open("classification.txt", "r")
for line in file:
    items = line.split(',')
    dataPoints.append([1, float(items[0]), float(items[1]), float(items[2])])

    if (items[4][0] == '+'):
        results.append(1)
    else:
        results.append(-1)

file.close()

(finalWeights,finalMissCount, finalRuntimes, finalMisclassifiedList, finalIterations) = pocket(dataPoints, results, 0.5, 7000)

print("Best weights " + str(finalWeights))
print("Accuracy: " + str(1-finalMissCount/len(dataPoints)))
# print("Run for " + str(finalRuntimes) + " times")
# print(finalMisclassifiedList)
# print(finalIterations)

# plt.plot(finalIterations, finalMisclassifiedList)
# plt.ylabel('#Misclassfied')
# plt.xlabel('#Iteration')
#
# x_new = np.linspace(1, len(finalIterations), 300)
# a_BSpline = interpolate.make_interp_spline(x, y)
# y_new = a_BSpline(x_new)
#
# plt.plot(x_new, y_new)
#
# plt.show()

xnew = np.linspace(0, len(finalIterations), len(finalIterations))

spl = make_interp_spline(finalIterations, finalMisclassifiedList, k=3)  # type: BSpline
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth)
plt.ylabel('#Misclassfied')
plt.xlabel('#Iteration')
plt.show()
