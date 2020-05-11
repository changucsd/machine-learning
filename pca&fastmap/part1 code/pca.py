import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import linalg as LA
import scipy.linalg as la

#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

# generate a sigma matrix using the mean and num of dim
def getCovariance(points, theMean, n):
    normalized = []

    for point in points:
        x = point[0] - theMean[0]
        y = point[1] - theMean[1]
        z = point[2] - theMean[2]
        normalized.append([x, y, z])

    normalized = np.array(normalized)
    dotProduct = np.dot(normalized.T, normalized)
    dotProduct = dotProduct * 1 / n
    return dotProduct


dataPoints = []

totalX = 0
totalY = 0
totalZ = 0

# read file:
file = open("pca-data.txt", "r")
for line in file:
    items = line.split('\t')

    totalX = totalX + float(items[0])
    totalY = totalY + float(items[1])
    totalZ = totalZ + float(items[2])
    dataPoints.append([float(items[0]), float(items[1]), float(items[2])])

file.close()

meanX = float(totalX / len(dataPoints))
meanY = float(totalY / len(dataPoints))
meanZ = float(totalZ / len(dataPoints))

means = [meanX, meanY, meanZ]

# get the coveraiance matrix
covarianceMatrix = getCovariance(dataPoints, means, len(dataPoints))
print (covarianceMatrix)

# get the sorted eigenvalues and eigenvectors

covarianceMatrix = np.array(covarianceMatrix)

eigenvalue = np.array
eigenvector = np.array
_v = np.array

# eigenvector, eigenvalue, _v = np.linalg.svd(covarianceMatrix)
eigenvalue,eigenvector = np.linalg.eig(covarianceMatrix)

k = 2
sorted_value_idx = np.argsort(-eigenvalue)  # Decreasing sort
sorted_eigenvector = eigenvector[:, sorted_value_idx]
sorted_eigenvalue = eigenvalue[sorted_value_idx]

sorted_eigenvector = eigenvector[sorted_value_idx]  # Get sorted eigenvalue
sorted_k_eigenvector = sorted_eigenvector[:, :k]

print (sorted_eigenvector)
print (sorted_eigenvalue)

npPoints =  np.loadtxt('pca-data.txt', delimiter='\t')
meanVals = np.mean(npPoints, axis=0)
meanRemoved = npPoints - meanVals

finalValue = np.dot(meanRemoved, sorted_k_eigenvector)
print(finalValue)

plt.scatter(finalValue[:, 0], finalValue[:, 1], marker='o')
plt.show()

