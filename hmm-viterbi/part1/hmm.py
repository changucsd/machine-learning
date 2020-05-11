#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#
import math
import numpy as np
import matplotlib.pyplot as plt


# check if (x2,y2) is next to (x1,x2)
# if so, check how many valid move direction can be made from (x1,y1), and get prob = 1/ (direction choice num)
# if not, return 0
def getProbs(x1, y1, x2, y2, grid):
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

    initial_prob = np.ones((1, valid_count)) * 1.0 / valid_count
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


class HMM(object):
    def __init__(self, cells, init_prob, transition_prob, emission_prob, observations, noisy_distance):
        self.cells = cells
        self.observations = observations  # 0.0 - 16.5
        self.init_prob = init_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob  # matrices of probability 165 * 87, 0.0 - 16.5 * 87, most of them is zero
        self.noisy_distance = noisy_distance
        self.traces = []
        return

    def viterbi(self):
        steps = len(self.noisy_distance)
        cells_max_prob = []

        for step in range(steps):
            # check the emission matrices to get a 1 * 87 table with values relevant to current nosies set
            emission_prob = self.getTowerTable(step)

            if step == 0:
                cells_max_prob = self.init_prob * emission_prob
            else:
                cells_max_prob, previous_cell = self.max_prob_cell(cells_max_prob * self.transition_prob)
                cells_max_prob = cells_max_prob * emission_prob
                self.traces.append(previous_cell)
        max_prob = -1
        max_i = -1
        for i in range(len(cells_max_prob[0])):
            if cells_max_prob[0][i] > max_prob:
                max_prob = cells_max_prob[0][i]
                max_i = i
        sequence = [max_i]
        self.traces.reverse()
        for trace in self.traces:
            sequence.append(trace[sequence[-1]])
        sequence.reverse()
        for i in range(len(sequence)):
            sequence[i] += 1
        return sequence

    def max_prob_cell(self, cell_prob):
        max_pre_state = 0
        max_probs_list = []
        max_pre_states_list = []
        for current_cell in range(87):  # Review each column from row1, ..., rowN.
            max_prob = 0
            for previous_state in range(87):
                prob = cell_prob[current_cell][previous_state]
                if prob > max_prob:
                    max_prob = prob
                    max_pre_state = previous_state
            max_probs_list.append(max_prob)
            max_pre_states_list.append(max_pre_state)
        return max_probs_list, max_pre_states_list

    def getTowerTable(self, step):
        tmp_cell_prob = []  # 4* 87
        cell_prob = np.ones((1, 87))
        # for each 4 tower, get the probabilities of 87 cells
        for i in range(len(self.noisy_distance[step])):
            tmp_cell_prob.append([])
            for j in range(len(self.cells)):
                for k in range(len(self.observations)):
                    if self.noisy_distance[step][i] == self.observations[k]:
                        tmp_cell_prob[i].append(self.emission_prob[i][j][k])
                        break
        # sum them up
        for _each_cell_prob in tmp_cell_prob:
            # 4 towers, each tower we calculate
            cell_prob *= _each_cell_prob
        return cell_prob


def rotate90Clockwise(A):
    N = len(A[0])
    for i in range(N // 2):
        for j in range(i, N - i - 1):
            temp = A[i][j]
            A[i][j] = A[N - 1 - j][i]
            A[N - 1 - j][i] = A[N - 1 - i][N - 1 - j]
            A[N - 1 - i][N - 1 - j] = A[j][N - 1 - i]
            A[j][N - 1 - i] = temp


def rotatePoint(results):
    output = []
    N = 10
    for point in results:
        temp = [point[1], N - 1 - point[0]]
        output.append(temp)
    return np.array(output)


if __name__ == "__main__":
    grid, tower_location, noises = load_file('hmm-data.txt')
    cells, valid_coordinates, initial_prob, transit_matrix, = read_grids(grid)
    emission_matrix, observations = getEmission(tower_location, cells, valid_coordinates)
    # print(cells)
    # print(initial_prob)
    # print(transit_matrix)
    # print(valid_coordinates)
    # print (observations)
    # print("noise")
    # print(noises)
    # print(emission_matrix)
    # print()

    hmm = HMM(cells=cells, init_prob=initial_prob, transition_prob=transit_matrix,
              emission_prob=emission_matrix, observations=observations, noisy_distance=noises)
    sequence = hmm.viterbi()
    print(sequence)

    result = []
    for i in sequence:
        result.append(valid_coordinates[i])
    print(result)

    # plot the trajectory on a graph
    grid = np.array(grid)
    rotate90Clockwise(grid)

    color = []
    print_x = []
    print_y = []
    for y in range(0, len(grid)):
        for x in range(0, len(grid)):

            print_x.append(x)
            print_y.append(y)
            if grid[x][y] == 0:
                color.append('red')
            else:
                color.append('white')

    plt.scatter(print_x, print_y, marker='o', c=color, edgecolors='black')
    result = np.array(result)
    result = rotatePoint(result)
    plt.scatter(result[:, 0], result[:, 1], marker='D', c='blue')

    step = 1
    plottedPoints = []
    for point in result:
        if point.tolist() in plottedPoints:
            plt.text(point[0] + 0.1, point[1] - 0.5, 'step ' + str(step))
        else:
            plt.text(point[0] + 0.1, point[1], 'step ' + str(step))
            plottedPoints.append(point.tolist())
        step += 1
    plt.show()
