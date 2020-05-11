from scipy.stats import multivariate_normal
from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np

import math
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import pyplot as plt
import numpy as np

#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

trials = 10
clusters = 3
X = np.genfromtxt(r'clusters.txt', delimiter=',')
print ('===================')
print ('     EM-GMM')
print ('===================')
gmm = mixture.GaussianMixture(n_components=clusters, n_init=trials, covariance_type="full")
gmm.fit(X)
labels = gmm.predict(X)
weights = gmm.weights_
means = gmm.means_
n_cov = gmm.covariances_


# colors = 2*["r.","g.","c.","b.","y."]
# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
#
# plt.scatter(means[:,0], means[:,1], marker='x', s=20, linewidths=10)
# plt.show()

print ('GMM weights:', weights)
print ('GMM means:', means)
print ('GMM covars: components=', n_cov)

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

X, Y = np.meshgrid(myX, myY)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

for index in range(0,len(means)):
    print("mu")
    print (means[index])
    print ("sigma")
    print(n_cov[index])
    rv = multivariate_normal(means[index], n_cov[index])

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

plt.show()
