import numpy as np

#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#


def logistic(data_list):
    data_list = np.array(data_list)
    row, col = data_list.shape
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    iterations = 0

    learning_rate = 0.01
    X = data_list[:,0:3]
    Y = data_list[:,3]
    bias_val = np.ones((row, 1))
    data = np.concatenate((bias_val, X), axis=1)
    while iterations < 7000:
        s = np.multiply((np.dot(data, weights)), Y)
        delta_Ein = np.sum((np.multiply(Y.T, data.T) / (1 + np.exp(s)).T).T, axis=0)
        delta_Ein = delta_Ein / len(data)
        v = (delta_Ein) / np.linalg.norm(delta_Ein)
        weights += (learning_rate * v)
        iterations = iterations + 1

    print(weights)
    print(get_error(data_list, weights))
    return


def get_error(data_list,weights):
    weights = np.array(weights)
    passNum = 0
    for data in data_list:
        label = data[3]
        data = data[:3]
        data = [1,data[0],data[1],data[2]]
        point = np.array(data)
        value = np.dot(weights.T, point)
        if value >= 0 and label == 1:
            passNum = passNum + 1
        elif value < 0 and label == -1:
            passNum = passNum + 1
    return (passNum / len(data_list))


def accuracyCalulator(weights, dataPoints,Ys):
    weights = np.array(weights)
    passNum = 0
    for i in range(0, len(dataPoints)):
        point = np.array(dataPoints[i])
        value = np.dot(weights.T, point)
        if value >= 0 and Ys[i] == 1:
            passNum = passNum + 1
        elif value < 0 and Ys[i] == -1:
            passNum = passNum + 1
    return (passNum / len(dataPoints))


data_list = []
with open("classification.txt", 'r') as in_file:
    for line in in_file.readlines():
        point = [float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2]),
                 float(line.split(",")[4])]
        data_list.append(point)
logistic(data_list)