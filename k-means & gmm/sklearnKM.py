import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

trials = 1
clusters = 3
X = np.genfromtxt(r'clusters.txt', delimiter=',')
print("X.shape=" + str(X.shape))
print('            K=Means')

km = KMeans(n_clusters= clusters, n_init=trials)
km.fit(X)

centroids = km.cluster_centers_
labels = km.labels_
plt.xlabel("x-axis")
plt.ylabel("y-axis")

colors = 2*["r.","g.","c.","b.","y."]

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]], markersize = 10)

plt.scatter(centroids[:,0],centroids[:,1], marker='x',linewidths= 8)
plt.show()
print('centroids=' + str(centroids), end='\n\n\n')