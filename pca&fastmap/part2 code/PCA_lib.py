#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.loadtxt("pca-data.txt")
pca = PCA(n_components=2)
pca.fit(data)
data_pca = pca.fit_transform(data)
print(data_pca)

plt.scatter(data_pca[:, 0], data_pca[:, 1], marker='o')
plt.show()


