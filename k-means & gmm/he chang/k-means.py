import math
import pandas as pd
import random
import sys
from matplotlib import pyplot as plt


def getDistance(x, y):
    a = float(x[0] - y[0])
    b = float(x[1] - y[1])
    return float(math.sqrt(a ** 2 + b ** 2))


def getCentroids(list):
    X = 0.0
    Y = 0.0

    for points in list:
        X = X + points[0]
        Y = Y + points[1]
    return [float(X / len(list)), float(Y / len(list))]


dataPoints = []
X = []
Y = []

# read file:
file = open("clusters.txt", "r")
for line in file:
    items = line.split(',')
    dataPoints.append([float(items[0]), float(items[1])])
    X.append((float(items[0])))
    Y.append((float(items[1])))

plt.scatter(X, Y)
plt.tight_layout()

centroids = []

for i in range(0, 3):
    randX = random.uniform(min(X), max(X))
    randY = random.uniform(min(Y), max(Y))
    centroids.append([randX, randY])
print("random centroids: ")
print(centroids)

stop = False
# [centroid] -> {(point),(point)....}
group = {}

while not stop:

    for point in dataPoints:
        minDistance = sys.float_info.max
        targetCentroid = []
        for centroid in centroids:
            distance = getDistance(point, centroid)
            if distance < minDistance:
                minDistance = distance
                targetCentroid = centroid.copy()

        key = str(targetCentroid[0]) + "," + str(targetCentroid[1])
        if key in group:
            group[key].append(point)
        else:
            group[key] = []
            group[key].append(point)
    print("new group formed:")
    print(group)

    newCentroids = []
    for list in group.values():
        newCentroids.append(getCentroids(list))
    print("new centroids set")
    print(newCentroids)

    if newCentroids[0] in centroids and newCentroids[1] in centroids and newCentroids[2] in centroids:
        stop = True
    else:
        centroids = newCentroids.copy()
        group.clear()

centX = []
centY = []
for centroid in centroids:
    centX.append(centroid[0])
    centY.append(centroid[1])

plt.xlabel("x-axis")
plt.ylabel("y-axis")

colors = 2*["r.","g.","c.","b.","y."]

plt.scatter(centX,centY, marker='x',linewidths= 8)
plt.show()


file.close()
