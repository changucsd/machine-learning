#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt


def SVM():
    H = np.zeros((len(Xs), len(Xs)))
    for i in range(len(Xs)):
        for j in range(len(Xs)):
            H[i, j] = Ys[i] * Ys[j] * np.dot(Xs[i], Xs[j])
    H = cvxopt_matrix(H)
    q = cvxopt_matrix(-1 * np.ones(len(Xs)))
    G = cvxopt_matrix(np.diag(np.ones(len(Xs)) * -1))
    h = cvxopt_matrix(np.zeros(len(Xs)))
    A = cvxopt_matrix(Ys, (1, len(Xs)))
    b = cvxopt_matrix(0.0)
    result = cvxopt_solvers.qp(H, q, G, h, A, b)
    alphas = np.array(result['x'])
    support_vector_indices = np.where(alphas > 0.0001)[0]
    print(support_vector_indices)
    alphas = alphas[support_vector_indices]
    support_vectors = Xs[support_vector_indices]
    support_vectors_y = Ys[support_vector_indices]
    print("%d support vectors out of %d points" % (len(alphas), len(Xs)))
    print(alphas)
    weights = np.zeros(2)
    for i in range(len(alphas)):
        weights += alphas[i] * support_vectors_y[i] * support_vectors[i]
    bias = Ys[support_vector_indices] - np.dot(Xs[support_vector_indices], weights)
    return weights, bias, support_vectors, support_vectors_y


if __name__ == "__main__":
    Xs = []
    Ys = []
    with open("linsep.txt", 'r') as in_file:
        for line in in_file.readlines():
            line = line.strip()
            x1 = float(line.split(",")[0])
            x2 = float(line.split(",")[1])
            Xs.append((x1, x2))
            Ys.append(float(line.split(",")[2]))
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    weights, bias, support_vectors, support_vectors_y = SVM()
    print(weights)
    print(bias)
    plt.scatter(Xs[:, 0], Xs[:, 1], c=Ys, cmap='bwr', alpha=1, s=50, edgecolors='k')
    x2_l = -(weights[0] * (-0.2) + bias) / weights[1]
    x2_r = -(weights[0] * (1) + bias) / weights[1]
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', s=100, edgecolors='k')
    plt.plot([-0.2, 1], [x2_l, x2_r])
    plt.show()
