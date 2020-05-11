#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

from sklearn.neural_network import MLPClassifier
import numpy as np
import imageio

N_FEATURES = 30 * 32
N_HIDDEN_SIZE = 100
weight_random_low = -0.01
weight_random_high = 0.01
TRAINING_SIZE = 184
TESTING_SIZE = 83
EPOCHS = 1000
LR = 0.1  # learning rate

TRAINING_LIST_NAME = 'downgesture_train.list'
TESTING_LIST_NAME = 'downgesture_test.list'


def getXandY(filename, sample_size):
    with open(filename) as file:
        training_list = file.read().splitlines()
    training_set_size = len(training_list)

    X = np.empty((0, N_FEATURES), float)

    for sample in training_list[:sample_size]:
        im = imageio.imread(sample) / 255.0
        X = np.vstack((X, im.flatten()))

    Y = np.zeros((training_set_size, 1))
    for i in range(training_set_size):
        if "down" in training_list[i]:
            Y[i] = 1
    Y = Y[:sample_size]
    return X, Y


X_train, Y_train = getXandY(TRAINING_LIST_NAME, TRAINING_SIZE)
X_test, Y_test = getXandY(TESTING_LIST_NAME, TESTING_SIZE)

mlp = MLPClassifier(solver='sgd', alpha=1e-15,
                    hidden_layer_sizes=(N_HIDDEN_SIZE,), activation='logistic', learning_rate_init=LR, max_iter=EPOCHS)

mlp.fit(X_train, Y_train.ravel())

test_filenames = []
with open('downgesture_test.list') as f:
    for training_image in f.readlines():
        test_filenames.append(training_image.strip())


for index in range(0, len(X_test)):

    x = np.array([X_test[index]])
    result = mlp.predict(x)
    if result >= 0.5:
        predicted_result = "is DOWN"
    else:
        predicted_result = "is Not DOWN"

    if predicted_result == "is DOWN" and 'down' in test_filenames[index]:
        check = " check"
    elif predicted_result == "is Not DOWN" and 'down' not in test_filenames[index]:
        check = " check"
    else:
        check = ""
    print(test_filenames[index] + ": " + predicted_result + check)

print("Accuracy for train:\n{a}".format(a=mlp.score(X_train, Y_train.ravel())))
print("Accuracy for test:\n{a}".format(a=mlp.score(X_test, Y_test.ravel())))
