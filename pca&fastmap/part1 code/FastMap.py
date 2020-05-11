import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D

#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#


def fastmap(iter):
    # recursion

    if iter > K:
        return
    # step 1 identified the fastest pair
    point_a = random.randint(1, NUM)
    point_b = -1
    old_point_a = -1
    ind = 0
    while ind < 6:
        max_d = 0
        for b in map[point_a]:
            if distance(iter, point_a, b) > max_d:
                point_b = b
                max_d = distance(iter, point_a, b)
        if old_point_a == point_b:
            break
        old_point_a = point_a
        point_a = point_b
        ind += 1

    distance_ab = distance(iter, point_a, point_b)
    # step 2 cal D(Oi, Oa) & D(Oi, Ob)
    # done by distance func

    # step 3 cal xi
    for point in range(1, NUM+1):
        if distance_ab == 0:
            xi = 0
        else:
            xi = math.pow(distance(iter, point_a, point), 2) + math.pow(distance(iter, point_b, point_a), 2) - math.pow(
                distance(iter, point_b, point), 2)
            xi /= 2 * distance_ab
        # if point in old_xi:
        #     old_xi[point][iter] = xi
        # else:
        #     old_xi[point] = {}
        #     old_xi[point][iter] = xi
        result_coodinate[point - 1].append(xi)
    fastmap(iter + 1)
    return

def distance(iter, point_a, point_b):
    #help calculate new distance
    if point_b == point_a:
        return 0
    if iter == 1:
        return map[point_a][point_b]
    return math.sqrt(math.pow(distance(iter - 1, point_a, point_b),2) - math.pow((result_coodinate[point_a-1][iter-2] - result_coodinate[point_b-1][iter-2]),2))

NUM = 10
K = 2 #dimension
map = {}
num_word_dict = {}

old_xi = {}
old_distance = []

result_coodinate = []
for i in range(1, NUM+1):
    result_coodinate.append([])

with open("fastmap-data.txt", 'r') as in_file:
    for line in in_file.readlines():
        a = int(line.split("\t")[0])
        b = int(line.split("\t")[1])
        dist = int(line.split("\t")[2])
        if a in map:
            map[a][b] = dist
        else:
            map[a] = {}
            map[a][b] = dist
        if b in map:
            map[b][a] = dist
        else:
            map[b] = {}
            map[b][a] = dist


def avg_distortion():
    if K == 2:
        #count 1.3318311090338195
        count = 0
        for i in range(1, NUM+1):
            for j in range(i + 1, NUM + 1):
                dist_ij = math.sqrt(math.pow((result_coodinate[i-1][0] - result_coodinate[j-1][0]) ,2)  + math.pow((result_coodinate[i-1][1] - result_coodinate[j-1][1]),2))
                count += abs(dist_ij - map[i][j])
        count /= NUM*(NUM-1)/2
        print("count")
        print(count)
    if K == 3:
        #count 0.7868935697305133
        count = 0
        for i in range(1, NUM + 1):
            for j in range(i + 1, NUM + 1):
                dist_ij = math.sqrt(math.pow((result_coodinate[i - 1][0] - result_coodinate[j - 1][0]), 2) + math.pow(
                    (result_coodinate[i - 1][1] - result_coodinate[j - 1][1]), 2)  + math.pow(
                    (result_coodinate[i - 1][2] - result_coodinate[j - 1][2]), 2))
                count += abs(dist_ij - map[i][j])
        count /= NUM * (NUM - 1) / 2
        print("count")
        print(count)
    if K == 4:
        #count 0.4063425242025564
        count = 0
        for i in range(1, NUM + 1):
            for j in range(i + 1, NUM + 1):
                dist_ij = math.sqrt(math.pow((result_coodinate[i - 1][0] - result_coodinate[j - 1][0]), 2) + math.pow(
                    (result_coodinate[i - 1][1] - result_coodinate[j - 1][1]), 2) + math.pow(
                    (result_coodinate[i - 1][2] - result_coodinate[j - 1][2]), 2) + math.pow(
                    (result_coodinate[i - 1][3] - result_coodinate[j - 1][3]), 2))
                count += abs(dist_ij - map[i][j])
        count /= NUM * (NUM - 1) / 2
        print("count")
        print(count)
    return

def show():
    index = 1
    fig, ax  = plt.subplots()
    for zi in result_coodinate:
        ax.scatter(zi[0], zi[1])
        ax.annotate(num_word_dict[index],(zi[0],zi[1]))
    # for zi in result_coodinate:
    #     plt.plot(zi[0], zi[1], 'r.')
        index += 1
    plt.show()


    # x = []
    # y = []
    # z = []
    # for zi in result_coodinate:
    #     x.append(zi[0])
    #     y.append(zi[1])
    #     z.append(zi[2])
    # ax = plt.subplot(111, projection='3d')

    # ax.scatter(x[:10], y[:10], z[:10], c='y')
    # ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
    # ax.scatter(x[30:40], y[30:40], z[30:40], c='g')
    #
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.show()
    return

with open("fastmap-wordlist.txt", 'r') as in_file:
    i = 1
    for line in in_file.readlines():
        num_word_dict[i] = line
        i += 1

fastmap(1)
for point in result_coodinate:
    print(point)
# avg_distortion()
show()
