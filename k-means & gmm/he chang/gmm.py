import math
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import pyplot as plt
import numpy as np

# function that generate normalized weights
def getRandomWeights(num):
    data = []
    totalValue = 0
    for n in range(0, num):
        valueR = random.uniform(0, 1)
        data.append(valueR)
        totalValue = totalValue + valueR
    scale = float(1 / totalValue)
    output = []
    for val in data:
        output.append(val * scale)
    return output

# generate a sigma matrix using the mean and weights
def getCovariance(points, theMean, weights):
    output = []
    totalWeight = float(0)

    # Nk
    for theWeight in weights:
        totalWeight = totalWeight + theWeight

    for index in range(0, len(points)):

        x = points[index][0] - theMean[0]
        y = points[index][1] - theMean[1]
        p = np.array([[x], [y]])
        cross = np.dot(p, p.T)
        cross = cross * weights[index]/totalWeight

        if len(output) == 0:
            output = cross
        else:
            output = np.add(output, cross)
    return output

# calculate gaussian model with the datapoint and rics
def getGaussians(theDataPoints, theClusters):
    # [(mu,sigma,pi),(mu,sigma,pi),(mu,sigma,pi)]
    gaussians = []

    for theCluster in theClusters:
        totalWeight = float(0)
        for theWeight in theCluster:
            totalWeight = totalWeight + theWeight

        meanX = 0
        meanY = 0
        for i in range(0, len(theDataPoints)):
            meanX = meanX + theDataPoints[i][0] * theCluster[i] / totalWeight
            meanY = meanY + theDataPoints[i][1] * theCluster[i] / totalWeight

        mu = [meanX, meanY]
        theMu = np.array([meanX, meanY])
        sigma = getCovariance(theDataPoints, mu, theCluster)

        pi = 0
        for var in theCluster:
            pi = pi + var
        pi = pi / len(theCluster)

        gaussians.append((theMu, sigma, pi))
    return gaussians

#
# def getLoglikelihood(gaussianGroups, data):
#     logVal = 0
#     for pair in data:
#         for gauss in gaussianGroups:
#             logVal = logVal + math.log(
#                 gauss[2] * multivariate_normal.pdf(pair, mean=gauss[0], cov=gauss[1]))
#     return logVal



dataPoints = []
myX = []
myY = []

# read file:
file = open("clusters.txt", "r")
for line in file:
    items = line.split(',')
    dataPoints.append([float(items[0]), float(items[1])])
    myX.append((float(items[0])))
    myY.append((float(items[1])))



stop = False

## [pointstring] - > [weights1,weight2,weight3]
mapData = {}

clusters = [[], [], []]

# assign random ric for each point with respect to each cluster
for point in dataPoints:
    key = str(point[0]) + "," + str(point[1])
    weight = getRandomWeights(3)
    mapData[key] = weight
    clusters[0].append(weight[0])
    clusters[1].append(weight[1])
    clusters[2].append(weight[2])

gaussianGroup = getGaussians(dataPoints, clusters)
# totalLogLike = getLoglikelihood(gaussianGroup, dataPoints)

print(gaussianGroup)

while not stop:

    newClusters = [[], [], []]
    for point in dataPoints:

        cluster = []
        # re-calculate ric for each point
        probs = []
        total = 0
        for g in gaussianGroup:
            prob = g[2] * multivariate_normal.pdf(point, mean=g[0], cov=g[1], allow_singular=True)
            # prob = two_d_gaussian(g[0], g[1], g[2], point)
            total = total + prob
            probs.append(prob)

        for i in range(0, 3):
            value = probs[i]
            ric = value / total
            newClusters[i].append(ric)

    currentgaussianGroup = getGaussians(dataPoints, newClusters)
    print("current Gaussians")
    for gauss in currentgaussianGroup:
        print(gauss)

    check1 = True
    check2 = True
    check3 = True

    for k in range(0, 3):

        old = gaussianGroup[k]
        new = currentgaussianGroup[k]

        if k == 0:
            check1 = abs(old[0][0] - new[0][0]) < 0.0001 and abs(old[0][1] - new[0][1]) < 0.0001

        elif k == 1:
            check2 = abs(old[0][0] - new[0][0]) < 0.0001 and abs(old[0][1] - new[0][1]) < 0.0001

        else:
            check3 = abs(old[0][0] - new[0][0]) < 0.0001 and abs(old[0][1] - new[0][1]) < 0.0001

    stop = check1 and check2 and check3
    gaussianGroup = currentgaussianGroup.copy()

X, Y = np.meshgrid(myX, myY)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y


print("final Gaussians")
for gauss in gaussianGroup:
    print("mu")
    print (gauss[0])
    print ("sigma")
    print(gauss[1])
    rv = multivariate_normal(gauss[0], gauss[1])

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

plt.show()
