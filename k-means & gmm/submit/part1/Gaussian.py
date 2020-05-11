import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D



#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#


def PDF(data, Mu, sigma):
    """
    Probablility Density Function
    """
    sigma_sqrt = math.sqrt(np.linalg.det(sigma))
    sigma_inv = np.linalg.inv(sigma)
    data.shape = (2, 1)
    Mu.shape = (2, 1)
    minus_mu = data - Mu
    minus_mu_trans = np.transpose(minus_mu)
    res = (1.0 / (2.0 * math.pi * sigma_sqrt)) * math.exp(
        (-0.5) * (np.dot(np.dot(minus_mu_trans, sigma_inv), minus_mu)))
    return res


def step_2(Data):
    """
    compute weighted means and variances
    """
    N_1 = 0
    N_2 = 0
    N_3 = 0

    for i in range(len(parameter["ri_1"])):
        N_1 = N_1 + parameter["ri_1"][i]
        N_2 = N_2 + parameter["ri_2"][i]
        N_3 = N_3 + parameter["ri_3"][i]
    # recompute mu
    new_mu_1 = np.array([0, 0])
    new_mu_2 = np.array([0, 0])
    new_mu_3 = np.array([0, 0])
    for i in range(len(parameter["ri_1"])):
        # new_mu_1 = new_mu_1 +  np.dot(parameter["ri_1"][i],Data[i]) / N_1
        # new_mu_2 = new_mu_2 + Data[i] * parameter["ri_2"][i] / N_2
        # new_mu_3 = new_mu_3 + Data[i] * parameter["ri_3"][i] / N_3
        new_mu_1 = new_mu_1 + np.dot(parameter["ri_1"][i], Data[i]) / N_1
        new_mu_2 = new_mu_2 + np.dot(parameter["ri_2"][i], Data[i]) / N_2
        new_mu_3 = new_mu_3 + np.dot(parameter["ri_3"][i], Data[i]) / N_3

    # numpy must define matrix shapes
    new_mu_1.shape = (2, 1)
    new_mu_2.shape = (2, 1)
    new_mu_3.shape = (2, 1)

    new_sigma_1 = np.array([[0, 0], [0, 0]])
    new_sigma_2 = np.array([[0, 0], [0, 0]])
    new_sigma_3 = np.array([[0, 0], [0, 0]])
    i = 0
    for point in Data:
        data_tmp = [0, 0]
        data_tmp[0] = point[0]
        data_tmp[1] = point[1]
        vec_tmp = np.array(data_tmp)
        vec_tmp.shape = (2, 1)
        new_sigma_1 = new_sigma_1 + np.dot(parameter["ri_1"][i], np.dot((vec_tmp - new_mu_1), (vec_tmp - new_mu_1).transpose()))
        new_sigma_2 = new_sigma_2 + np.dot(parameter["ri_2"][i], np.dot((vec_tmp - new_mu_2), (vec_tmp - new_mu_2).transpose()))
        new_sigma_3 = new_sigma_3 + np.dot(parameter["ri_3"][i], np.dot((vec_tmp - new_mu_3), (vec_tmp - new_mu_3).transpose()))
        i += 1
    new_sigma_1 = new_sigma_1 / N_1
    new_sigma_2 = new_sigma_2 / N_2
    new_sigma_3 = new_sigma_3 / N_3
    new_pi_1 = N_1 / len(parameter["ri_1"])
    new_pi_2 = N_2 / len(parameter["ri_2"])
    new_pi_3 = N_3 / len(parameter["ri_3"])

    # update all parameter
    parameter["Mu_1"] = new_mu_1
    parameter["Mu_2"] = new_mu_2
    parameter["Mu_3"] = new_mu_3
    parameter["Sigma_1"] = new_sigma_1
    parameter["Sigma_2"] = new_sigma_2
    parameter["Sigma_3"] = new_sigma_3
    parameter["Pi_weight_1"] = new_pi_1
    parameter["Pi_weight_2"] = new_pi_2
    parameter["Pi_weight_3"] = new_pi_3


def step_3(Data):
    """
    step 3: recompute responsibilities ric
    """

    sigma_1 = parameter["Sigma_1"]
    sigma_2 = parameter["Sigma_2"]
    sigma_3 = parameter["Sigma_3"]
    pw_1 = parameter["Pi_weight_1"]
    pw_2 = parameter["Pi_weight_2"]
    pw_3 = parameter["Pi_weight_3"]
    mu_1 = parameter["Mu_1"]
    mu_2 = parameter["Mu_2"]
    mu_3 = parameter["Mu_3"]

    parameter["ri_1"] = []
    parameter["ri_2"] = []
    parameter["ri_3"] = []

    for point in Data:
        pdf1 = pw_1 * PDF(point, mu_1, sigma_1)
        pdf2 = pw_2 * PDF(point, mu_2, sigma_2)
        pdf3 = pw_3 * PDF(point, mu_3, sigma_3)
        down = pdf1 + pdf2 + pdf3
        ri1 = pdf1 / down
        ri2 = pdf2 / down
        ri3 = pdf3 / down
        parameter["ri_1"].append(ri1)
        parameter["ri_2"].append(ri2)
        parameter["ri_3"].append(ri3)


def iterate(Data, esp=0.00001):
    while (True):
        old_mu_1 = parameter["Mu_1"].copy()
        old_mu_2 = parameter["Mu_2"].copy()
        old_mu_3 = parameter["Mu_3"].copy()
        step_2(Data)
        step_3(Data)
        delta_1 = parameter["Mu_1"] - old_mu_1
        delta_2 = parameter["Mu_2"] - old_mu_2
        delta_3 = parameter["Mu_3"] - old_mu_3


        if math.fabs(delta_1[0][0]) <= esp and math.fabs(delta_1[1][0]) <= esp and math.fabs(
                delta_2[0][0]) <= esp and math.fabs(delta_2[1][0]) <= esp and math.fabs(
                delta_3[0][0]) <= esp and math.fabs(delta_3[1][0]) <= esp:
            break
    print("====================")

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

    print("Result:")
    print("Mu_1: ")
    print(parameter["Mu_1"])
    print("Sigma_1: ")
    print(parameter["Sigma_1"])
    print("Pi_1: ")
    print(parameter["Pi_weight_1"])

    meanX = parameter["Mu_1"][0][0]
    meanY = parameter["Mu_1"][1][0]

    print (np.array([meanX,meanY]))
    rv = multivariate_normal(np.array([meanX,meanY]), parameter["Sigma_1"])
    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


    print("Mu_2: ")
    print(parameter["Mu_2"])
    print("Sigma_2: ")
    print(parameter["Sigma_2"])
    print("Pi_2: ")
    print(parameter["Pi_weight_2"])

    meanX = parameter["Mu_2"][0][0]
    meanY = parameter["Mu_2"][1][0]

    print (np.array([meanX,meanY]))
    rv = multivariate_normal(np.array([meanX,meanY]), parameter["Sigma_2"])
    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    print("Mu_3: ")
    print(parameter["Mu_3"])
    print("Sigma_3: ")
    print(parameter["Sigma_3"])
    print("Pi_3: ")
    print(parameter["Pi_weight_3"])

    meanX = parameter["Mu_3"][0][0]
    meanY = parameter["Mu_3"][1][0]

    print (np.array([meanX,meanY]))
    rv = multivariate_normal(np.array([meanX,meanY]), parameter["Sigma_3"])
    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

def start_GMM():
    Data_list = []
    with open("clusters.txt", 'r') as in_file:
        for line in in_file.readlines():
            point = []
            point.append(float(line.split(",")[0]))
            point.append(float(line.split(",")[1]))
            Data_list.append(point)
    Data = np.array(Data_list)
    # Assign ric randomly, K=3
    for p in Data:
        r1 = random.randint(1,50)
        r2 = random.randint(1,50)
        r3 = random.randint(1,50)
        #normalize
        ri1 = r1 / (r1 + r2 + r3)
        ri2 = r2 / (r1 + r2 + r3)
        ri3 = r3 / (r1 + r2 + r3)
        parameter["ri_1"].append(ri1)
        parameter["ri_2"].append(ri2)
        parameter["ri_3"].append(ri3)
    # start iteration
    iterate(Data)


if __name__ == '__main__':
    parameter = {}
    parameter["Mu_1"] = np.array([0, 0])
    parameter["Sigma_1"] = np.array([[1, 0], [0, 1]])
    parameter["Mu_2"] = np.array([0, 0])
    parameter["Sigma_2"] = np.array([[1, 0], [0, 1]])
    parameter["Mu_3"] = np.array([0, 0])
    parameter["Sigma_3"] = np.array([[1, 0], [0, 1]])
    parameter["Pi_weight_1"] = 0.5
    parameter["Pi_weight_2"] = 0.5
    parameter["Pi_weight_3"] = 0.5
    parameter["ri_1"] = []
    parameter["ri_2"] = []
    parameter["ri_3"] = []
    start_GMM()
