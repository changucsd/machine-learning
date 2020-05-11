#
# Authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

import numpy as np


def read_pgm(pgm):
    with open(pgm, 'rb') as f:
        f.readline()  # skip secret word
        f.readline()  # skip comments
        xs, ys = f.readline().split()
        xs = int(xs)
        ys = int(ys)
        max_scale = int(f.readline().strip())

        image = [1]  # add bia valued 1 for each image
        for _ in range(xs * ys):
            image.append(f.read(1)[0] / max_scale)

        return image


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def feed_forward(dataPoints, W1, W2):
    # dataPoints in N*d shape => need d*N
    S1 = np.dot(W1, dataPoints.T)
    X1 = []
    for element in S1:
        list = []
        for s in element:
            list.append(sigmoid(s))
        X1.append(list)
    X1 = np.array(X1)

    S2 = np.dot(W2, X1)
    X2 = []
    for element in S2:
        list = []
        for s in element:
            list.append(sigmoid(s))
        X2.append(list)
    X2 = np.array(X2)

    return X1, X2


def back_propagate(dataPoints, Ys, W1, W2, X1, Y_hat, learningRate):
    logistic_derivation = lambda logistic_x: logistic_x * (1.0 - logistic_x)

    # W2 is the weights for layer 2, output layer
    # update W2

    # error_term_derivation = 2 * (Y_hat - Ys.T)  # 1*N
    #
    # dZ2 is the derivative for cost func in respect to Z2, Z2 is the input to layer 2
    # dZ2 = []  # dZ2 = 2(Y_hat - Ys) * (Y_hat * (1-Y_hat)) -- 1*N
    #

    activation_derivation = []  # 1 * N
    for element in Y_hat:
        activation_derivation.append([logistic_derivation(s) for s in element])

    error_term_derivation = 2 * (Y_hat - Ys.T)  # 1*N

    dZ2 = np.multiply(activation_derivation,error_term_derivation) # dZ2 = 2(Y_hat - Ys) * (Y_hat * (1-Y_hat)) -- 1*N
    dZ2 = np.array(dZ2)

    # dW2 is the derivative for cost func in respect to W2, W2 is the weights of layer 2
    dW2 = np.dot(dZ2, X1.T)  # dW2 = X1.T * dZ2 -- 1*1000

    # W1 is the weights for layer 1, hidden layer
    # update W1
    activation_derivation = []
    for element in X1:
        activation_derivation.append([logistic_derivation(s) for s in element])

    # dZ1 is the derivative for cost func in respect to Z1, Z1 is the input to layer 1
    dZ1 = []  # dZ1 = dZ2 * W2.T * (X1 * (1-X1)) -- 1000 * N
    W2dZ2 = np.dot(W2.T, dZ2)

    # 1000*N element-wise product
    dZ1 = np.multiply(W2dZ2, activation_derivation)

    dZ1 = np.array(dZ1)

    # dW1 is the derivative for cost func in respect to W1, W1 is the weights of layer 1
    dW1 = np.dot(dZ1, dataPoints)  # dW1 = dZ1 * X0.T --1000 * d

    W2 = W2 - learningRate * dW2  # W2 = W2 - learning_rate * dw2
    W1 = W1 - learningRate * dW1  # W1 = W1 - learn_rate * dW1

    return W1, W2


def getAccuracy(W1, W2, data, labels, filenames):
    passNum = 0

    X1, Y_hat = feed_forward(data, W1, W2)
    # print (X1)

    Y_hat = Y_hat.T


    # print (Y_hat)
    # print (labels)
    for index in range(0, len(labels)):

        result = Y_hat[index][0]
        label = labels[index][0]
        if result >=  0.5:
            predicted_result = "is DOWN"
        else:
            predicted_result = "is Not DOWN"

        print(filenames[index] + ": " + predicted_result)
        # print (result)
        # print (label)
        # print (result)
        # print (label)
        if result >= 0.5 and label == 1:
            passNum = passNum + 1
        elif result < 0.5 and label == 0:
            passNum = passNum + 1

    return float(passNum / len(data))


def neural_network(dataPoints, Ys, layer_size, epochs, learning_rate):
    dataDim = len(dataPoints[0])
    # N*dim shape

    # W1 is the weights for layer 1, hidden layer
    # W2 is the weights for layer 2, output layer
    W1 = np.random.uniform(-1, 1, size=(layer_size, dataDim)) * 0.01
    W2 = np.random.uniform(-1, 1, size=(1, layer_size)) * 0.01

    # Execute training.
    for i in range(0, epochs):

        print (i)
        # select sample data and labels

        # mini-batch, pick N random instances
        N = np.random.randint(1, len(dataPoints) + 1)  # N instances
        idx = np.random.choice(dataPoints.shape[0], N, replace=False)
        dataSamples = dataPoints[idx]
        labelSamples = Ys[idx]

        # round robin through all N instances
        for index in range(0, len(dataSamples)):
            # feed forward
            inputData = np.array([dataSamples[index]])
            label = np.array([labelSamples[index]])
            X1, Y_hat = feed_forward(inputData, W1, W2)
            # back propagation
            W1, W2 = back_propagate(inputData, label, W1, W2, X1, Y_hat, learning_rate)
    return W1, W2


images = []
labels = []

with open('downgesture_train.list') as f:
    for training_image in f.readlines():
        training_image = training_image.strip()
        images.append(read_pgm(training_image))
        if 'down' in training_image:
            labels.append([1, ])
        else:
            labels.append([0, ])

images = np.array(images)
labels = np.array(labels)

NNW1, NNW2 = neural_network(images, labels, layer_size=100, epochs=1000, learning_rate=0.1)

# predict on the testing data
testImages = []
testLabels = []
test_filenames = []
with open('downgesture_test.list') as f:
    for training_image in f.readlines():
        training_image = training_image.strip()
        testImages.append(read_pgm(training_image))
        test_filenames.append(training_image)
        if 'down' in training_image:
            testLabels.append([1, ])
        else:
            testLabels.append([0, ])

testImages = np.array(testImages)
testLabels = np.array(testLabels)

print("Accuracy: " + str(getAccuracy(NNW1, NNW2, testImages, testLabels, test_filenames)))