import math
import random
import matplotlib.pyplot as plt

#
# authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

def cal_distance(point1, point2):
    x = math.pow((point1[0] - point2[0]),2)
    y = math.pow((point1[1] - point2[1]),2)
    return math.sqrt(x+y)

# For each point P, assign it to closest centroid
def step_2(Data_list):
    parameter_dict["membership"] = [set(),set(),set()]
    for point in range(len(Data_list)):
        # assign centroid for each point
        min_d = 100000
        next_centroid_id = 0
        id = 0
        for centroid in parameter_dict["centroids"]:
            dist = cal_distance(Data_list[point], centroid)
            if dist < min_d:
                next_centroid_id = id
                min_d = dist
            id += 1
        parameter_dict["membership"][next_centroid_id].add(point)
    return

# Recompute centroid for each cluster
def step_3(Data_list):
    parameter_dict["centroids"] = []
    for membership in parameter_dict["membership"]:
        total_x = 0
        total_y = 0
        length = len(membership)
        if length == 0:
            parameter_dict["centroids"].append(Data_list[random.randint(0,149)])
            continue
        for point in membership:
            total_x += Data_list[point][0]
            total_y += Data_list[point][1]
        new_centroid = [total_x/length, total_y/length]
        parameter_dict["centroids"].append(new_centroid)
    return

def membership_same(old_memberships, new_memberships):
    for index in range(len(old_memberships)):
        if old_memberships[index] != new_memberships[index]:
            return False
    return True

def startKMeans():
    # read data
    Data_list = []
    with open("clusters.txt", 'r') as in_file:
        for line in in_file.readlines():
            point = []
            point.append(float(line.split(",")[0]))
            point.append(float(line.split(",")[1]))
            Data_list.append(point)
    # Randomly pick K centroids, K = 3
    for i in range(3):
        index = random.randint(0, len(Data_list)-1)
        parameter_dict["centroids"].append(Data_list[index])
    while True:
        old_memberships = parameter_dict["membership"].copy()
        step_2(Data_list)
        step_3(Data_list)
        if membership_same(old_memberships, parameter_dict["membership"]):
            break
    print(parameter_dict["centroids"])
    Show(Data_list)

def Show(data):
    num = 150
    dim = 2
    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
    k = 3
    cluster = parameter_dict["membership"]
    cp = parameter_dict["centroids"]
    ##二维图
    if dim == 2:
        mark = 0
        for member in cluster:
            for point in member:
                d = data[point]
                plt.plot(d[0], d[1], color[mark] + 'o')
            mark += 1

        # for i in num:
        #     mark = int(cluster[i, 0])
        #     plt.plot(data[i, 0], data[i, 1], color[mark] + 'o')

        for i in range(k):
            plt.plot(cp[i][0], cp[i][1], color[i] + 'x')
    plt.show()


if __name__ == '__main__':
    parameter_dict = {}
    parameter_dict["centroids"] = []
    parameter_dict["membership"] = [set(),set(),set()] #data_number
    startKMeans()