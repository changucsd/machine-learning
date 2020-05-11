#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#
import math
import numpy as np

# from hmmlearn import hmm
# import hidden_markov
# import hmms


def getProbs(x1, y1, x2, y2, grid):
    # check if (x2,y2) is next to (x1,x2)
    # if so, check how many valid move direction can be made from (x1,y1), and get prob = 1/ (direction choice num)
    # if not, return 0
    if ((y2 + 1 == y1) and (x2 == x1)) or ((x2 + 1 == x1) and (y2 == y1)) or ((y2 - 1 == y1) and (x2 == x1)) or (
            (x2 - 1 == x1) and (y2 == y1)):

        count = 0
        if (y1 - 1 >= 0) and (grid[x1][y1 - 1] == 1):
            count += 1
        if (x1 - 1 >= 0) and (grid[x1 - 1][y1] == 1):
            count += 1
        if (x1 + 1 < 10) and (grid[x1 + 1][y1] == 1):
            count += 1
        if (y1 + 1 < 10) and (grid[x1][y1 + 1] == 1):
            count += 1
        return 1.0 / count
    else:
        return 0.0


def load_file(file_name):
    grid = []
    towers = []
    noises = []
    with open(file_name) as f:
        f.readline()
        f.readline()
        # read gird information
        for i in range(0, 10):
            grid.append([int(value) for value in f.readline().split()])
        # skip lines
        for i in range(4):
            f.readline()
        # read tower information
        for i in range(0, 4):
            towers.append([int(value) for value in f.readline().split()[2:]])
        # skip lines
        for i in range(4):
            f.readline()
        # read noises
        for i in range(0, 11):
            noises.append([float(value) for value in f.readline().split()])

        return grid, towers, noises


def read_grids(grid):
    states = []  # a list of ids for valid cells
    valid_coordinates = {}  # a dictionary to translate valid state id to its actual coordinates

    # calculate the valid number of cells for states and get probabilities for initial_prob
    x = 0
    y = 0
    total_count = 0
    valid_count = 0
    for row in grid:
        for col in row:
            if col == 1:
                valid_coordinates[(valid_count + 1)] = [x, y]
                states.extend([valid_count + 1])
                valid_count += 1
            total_count += 1
            y += 1
        y = 0
        x += 1

    initial_prob = np.ones((1, valid_count)) * 1 / valid_count
    transit_matrix = np.zeros((valid_count, valid_count))

    # loop through the grid and check the possibilities of going from one cell to another within one step
    for prev in range(valid_count):
        for now in range(valid_count):
            x_prev, y_prev = valid_coordinates[prev + 1]
            x_now, y_now = valid_coordinates[now + 1]
            transit_matrix[prev][now] = getProbs(x_prev, y_prev, x_now, y_now, grid)

    return states, valid_coordinates, initial_prob, transit_matrix,


def getEmission(towers, states, valid_coordinates):
    emission_matrix = []
    observations = []

    max_distance = math.sqrt(9 ** 2 + 9 ** 2)

    for i in range(0, int(max_distance * 1.3 / 0.1) + 1):
        observations.extend([round(i * 0.1, 1)])

    for tower in towers:
        local_prob = []
        for state in states:
            d = math.sqrt((valid_coordinates[state][0] - tower[0]) ** 2 + (valid_coordinates[state][1] - tower[1]) ** 2)

            min_d = round(0.7 * d, 1)
            max_d = round(1.3 * d, 1)

            if max_d - min_d == 0:
                prob = 1
            else:
                prob = 1 / ((max_d - min_d) * 10)
            local_prob.append([])
            for object in observations:
                if min_d <= object <= max_d:
                    local_prob[state - 1].extend([prob])
                else:
                    local_prob[state - 1].extend([0])
        emission_matrix.append(local_prob)
    return emission_matrix, observations


grid, tower_location, noises = load_file('hmm-data.txt')
cells, valid_coordinates, initial_prob, transit_matrix, = read_grids(grid)
emission_matrix, observations = getEmission(tower_location, cells, valid_coordinates)

# print(cells)
# print(initial_prob)
# print(np.array(initial_prob[0]).shape)
# print(transit_matrix)
# print(np.array(transit_matrix).shape)
# print(valid_coordinates)
# print (observations)
# print("noise")
# print(noises)
# print(np.array(noises).shape)
# print(emission_matrix)
# print(np.array(emission_matrix).shape)

################### hmm ###############################
# n_states = len(cells)
# model = hmm.MultinomialHMM(n_components=n_states)
# model.startprob_ = np.array(initial_prob[0])
# model.transmat_ = np.array(transit_matrix)
# model.emissionprob_ = np.array(emission_matrix[0])
#
# robot_observation = np.array([[1, 2, 4, 6]]).T
# steps = len(noises)
#
# X, Z = model.sample(11)
#
# print (X)

# logprob, results = model.decode(robot_observation, algorithm="viterbi")
#
# robot_observation = robot_observation.T
# print (robot_observation.tolist())
#
# print (results)
# print("The observation is:", ", ".join(map(lambda x: observations[x], robot_observation[0].tolist())))
# print("The hidden states are:", ", ".join(map(lambda x: cells[x], results)))

################## hidden_markov.hmm #############################

# Initialize class object
# model = hidden_markov.hmm(cells,observations,np.array(initial_prob[0]),np.array(transit_matrix),np.array(emission_matrix[0]))
# robot_observation = np.array([[1, 2, 4, 6]]).T
# print(model.viterbi(robot_observation))

################## hmms.DtHMM #############################
# Create DtHMM by given parameters.
# dhmm = hmms.DtHMM(np.array(transit_matrix),np.array(emission_matrix[0]),np.array(initial_prob[0]))
