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


def kernel_func(x, y):
    # func(x1, x2)=(1 + x1 + x2) ^ 2
    return (1 + np.dot(x, y)) ** 2


def SVM_soft():
    H = np.zeros((len(Xs), len(Xs)))
    for i in range(len(Xs)):
        for j in range(len(Xs)):
            H[i, j] = Ys[i] * Ys[j] * kernel_func(Xs[i], Xs[j])

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
    sum = 0
    for i in range(len(support_vector_indices)):
        sum += alphas[i] * support_vectors_y[i] * kernel_func(support_vectors[i], support_vectors[support_vector_indices[0]])
    bias = Ys[support_vector_indices[0]] - sum
    return alphas, bias, support_vectors, support_vectors_y, support_vector_indices


if __name__ == "__main__":
    Xs = []
    Ys = []
    with open("nonlinsep.txt", 'r') as in_file:
        for line in in_file.readlines():
            line = line.strip()
            x1 = float(line.split(",")[0])
            x2 = float(line.split(",")[1])
            Xs.append((x1, x2))
            Ys.append(float(line.split(",")[2]))
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    alphas, bias, support_vectors, support_vectors_y, support_vector_indices = SVM_soft()
    print ('support vectors:')
    print (support_vectors)
    print("Intercept:")
    print(bias)
    # Create the contour hyperplane

    x_min, x_max = Xs[:, 0].min() - 1, Xs[:, 0].max() + 1
    y_min, y_max = Xs[:, 1].min() - 1, Xs[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

    z = []
    for point in np.c_[np.c_[xx.ravel(), yy.ravel()]]:
        # print (point)
        # transit to point in Z
        #
        WX = 0
        for i in range(len(support_vector_indices)):
            WX += alphas[i] * support_vectors_y[i] * kernel_func(support_vectors[i],point)
        result = np.sign(WX + bias)
        z.append(result)
    z = np.array(z)
    print(z)
    z = z.reshape(xx.shape)
    color = ['orange' if c == -1 else 'black' for c in Ys]

    plt.scatter(Xs[:, 0], Xs[:, 1], c=color)
    plt.scatter(Xs[support_vector_indices][:, 0], Xs[support_vector_indices][:, 1], c='red', marker='^')

    plt.contour(xx, yy, z)
    plt.show()
