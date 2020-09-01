# Elad Israel 313448888

import sys
import numpy as np
from numpy import linalg as lin

# Normalization of matrix's values to between newMin and newMax
def MinMaxNormalization(arr, newMin, newMax):
    columnsNum = len(arr[0])
    # calculate min and max values in each column(out of all the values in that column) and put it in the arrays.
    minInColumns = np.min(arr, axis=0)
    maxInColumns = np.max(arr, axis=0)
    for row in range(len(arr)):
        for column in range(columnsNum):
            # avoid division by 0
            if (maxInColumns[column] - minInColumns[column] == 0):
                break
            arr[row][column] = ((arr[row][column] - minInColumns[column]) / (
                maxInColumns[column] - minInColumns[column])) * (newMax - newMin) + newMin
    return arr


# go through each column(axis=0) and zscore it, returns the updated martix
def ZScoreNormalization(matrix):
    return (matrix - matrix.mean()) / matrix.std()


"""
Passive Aggressive multiclass training algorithm:
Online SVM version(getting sample after sample) and using tau instead of eta-> different variable which minimizes
 our W(as calculated in class)
"""
def PATrainingAlg(trainingSamplesX, trainingLabelsY, epochs):
    # number of columns in each row(number of features)
    featuresNum = len(trainingSamplesX[0])
    # create vector w with 3 rows- one for each class, and 8 columns - one for each feature(columns)
    w = np.asarray(np.zeros((3, featuresNum)))

    for e in range(epochs):
        # for each tuple:(training row, training true label)
        for x, y in zip(trainingSamplesX, trainingLabelsY):
            # prediction- multiply x with every row in W and the biggest result is x's predicted classification.
            y_hat = np.argmax(np.dot(w, x))
            # loss is 1- W at the row of the correct class * x, plus W at the row of the falsly predicted class * x
            loss = max(0, 1 - np.dot(w[y, :], x) + np.dot(w[y_hat, :], x))
            normX = lin.norm(x)
            # avoid division by 0
            if normX == 0:
                normX = 0.00001
            # tau according to what we calculated in class: loss/(2*(||x||^2)
            tau = loss / (2 * (normX ** 2))

            # update if classifier was wrong
            if y != y_hat:
                # raise the weight of the true classification of x by adding x*tau to W at the true label's row.
                w[y, :] = w[y, :] + tau * x
                # lower the weight of the false predicted class of x by subtracting x*tau from W at the falsely predicted class's row.
                w[y_hat, :] = w[y_hat, :] - tau * x
    return w


"""
SVM multiclass training algorithm:
Same as Perceptron, except now we're also trying to avoid overfitting by keeping margins at each side of the separator line.
SVM tries to maximize the "support vector" - the distance between two closest opposite sample points.
here we multiply each row by (1 - eta * lamda) 
"""
def SVMTrainingAlg(trainingSamplesX, trainingLabelsY, epochs, eta, lamda):
    # number of columns in each row(number of features)
    featuresNum = len(trainingSamplesX[0])
    # create vector w with 3 rows- one for each class, and 8 columns - one for each feature(columns)
    w = np.asarray(np.zeros((3, featuresNum)))

    for e in range(epochs):
        # for each tuple:(training row, training true label)
        for x, y in zip(trainingSamplesX, trainingLabelsY):
            # prediction- multiply x with every row in W and the biggest result is x's predicted classification.
            y_hat = np.argmax(np.dot(w, x))
            # update if classifier was wrong
            if y != y_hat:
                # raise the weight of the true classification of x by adding x*eta to W at the true label's row.
                w[y, :] = (1 - eta * lamda) * w[y, :] + eta * x
                # lower the weight of the false predicted class of x by subtracting x*eta from W at the falsely predicted class's row.
                w[y_hat, :] = (1 - eta * lamda) * w[y_hat, :] - eta * x
                # figure out which row is left unchanged by subtracting changed indexes from the sum of all indexes.
                unchangedRow = 3 - y - y_hat
                # all other rows should remain the same, except for margin changes.
                w[unchangedRow, :] = (1 - eta * lamda) * w[unchangedRow, :]
    return w


"""
Perceptron multiclass training algorithm:
1. Define W = matrix of 3 rows- one for each class and 8 columns - one for each feature(represent that feature's weight in that class).
2. Now we will update W 'epochs' times:
    2.1. for each tuple:(training row, training true label)
        2.1.1. the prediction(y_hat) is the index of the row in W(class) which its projection is the biggest.
                meaning- we multiply x with every row in W and the biggest result is x's predicted classification.
        2.1.2. if the prediction wasn't right(for example the index was 1 and we predicted 2):
            2.1.2.1. we will raise the weight of the true classification of x by adding x*eta to W at the true label's row.
                     and lower the weight of the false predicted class of x by subtracting x*eta from W at the falsely predicted class's row.
                     all other rows should remain the same.
3. return W - a multiclass classifier. Assuming we trained W right, if we will later multiply x with every row in W 
    we should get the correct classification(true label) of x, even on new samples(testing samples) most of the time.
"""
def perceptronTrainingAlg(trainingSamplesX, trainingLabelsY, epochs, eta):
    # number of columns in each row(number of features)
    featuresNum = len(trainingSamplesX[0])
    # create vector w with 3 rows- one for each class, and 8 columns - one for each feature(columns)
    w = np.asarray(np.zeros((3, featuresNum)))

    for e in range(epochs):
        # for each tuple:(training row, training true label)
        for x, y in zip(trainingSamplesX, trainingLabelsY):
            # prediction- multiply x with every row in W and the biggest result is x's predicted classification.
            y_hat = np.argmax(np.dot(w, x))
            # update if classifier was wrong
            if y != y_hat:
                # raise the weight of the true classification of x by adding x*eta to W at the true label's row.
                w[y, :] = w[y, :] + eta * x
                # lower the weight of the false predicted class of x by subtracting x*eta from W at the falsely predicted class's row.
                w[y_hat, :] = w[y_hat, :] - eta * x
    return w


# Test W on the testing samples and return success percentages
def testWOnTestingSamples(classPredictions, testingLabelsY):
    # number of rows(number of testing samples)
    testingSamplesNum = len(testingLabelsY)
    if testingSamplesNum == 0:
        return -1
    fails = 0
    for y, y_hat in zip(testingLabelsY, classPredictions):
        # wrong prediction
        if y != y_hat:
            fails += 1
    # return success percentage
    return (1 - (float(fails) / testingSamplesNum)) * 100


# classify the samples using the provided W
def classifyWithW(testingSamplesX, w):
    classPredictions = []
    for x in testingSamplesX:
        # prediction- multiply x with every row in W and the biggest result is x's predicted classification.
        classPredictions.append(np.argmax(np.dot(w, x)))
    return classPredictions


# one hot encoding
def convertSexStrToVector(sex):
    m = [0, 0, 1]
    f = [0, 1, 0]
    i = [1, 0, 0]
    if sex == "M":
        return np.asarray(m).astype(np.float)
    elif sex == "F":
        return np.asarray(f).astype(np.float)
    elif sex == "I":
        return np.asarray(i).astype(np.float)


# converts the "Sex" column(1st column) to numerical values, and the whole array to float
def convertToNumerical(data):
    firstColumn = data[:, 0]
    # convert each value in the first column to an appropriate vector
    firstColumn = np.asarray([convertSexStrToVector(s) for s in firstColumn])
    # delete first column from the array
    data = data[:, 1:].astype(np.float)
    # add the created vectors as 3 columns in the array instead of the original column
    data = np.asarray([np.concatenate((s, f), axis=None) for s, f in zip(firstColumn, data)])
    # return new array as float
    return data.astype(np.float)


def main():
    if len(sys.argv) < 4:
        print("error")
        exit(-1)
    trainingX = np.loadtxt(sys.argv[1], delimiter=',', dtype="str")
    trainingY = np.loadtxt(sys.argv[2], delimiter=',', dtype="int")
    testingSamples = np.loadtxt(sys.argv[3], delimiter=',', dtype="str")

    trainingX = convertToNumerical(trainingX)
    testingSamples = convertToNumerical(testingSamples)

    # shuffle, seed maked the random values the same on every run
    indices = np.arange(trainingX.shape[0])
    np.random.seed(1)
    np.random.shuffle(indices)
    trainingX = trainingX[indices]
    trainingY = trainingY[indices]

    # normalization
    # trainingX = MinMaxNormalization(trainingX,0,1)
    # trainingX = ZScoreNormalization(trainingX)

    # number of rows(number of samples)
    samplesNum = len(trainingX)
    # split to 80% training set and 20% testing set
    trainingSamplesX, trainingTestSamplesX = np.split(trainingX, [int(samplesNum * 0.8)])
    trainingLabelsY, trainingTestLablesY = np.split(trainingY, [int(samplesNum * 0.8)])
    epochs = 30
    eta = 0.1
    # training
    wPerceptron = perceptronTrainingAlg(trainingSamplesX, trainingLabelsY, epochs, eta)
    eta = 0.1
    lamda = 0.0001
    wSVM = SVMTrainingAlg(trainingSamplesX, trainingLabelsY, epochs, eta, lamda)
    wPA = PATrainingAlg(trainingSamplesX, trainingLabelsY, epochs)

    """ on comments- algorithms already tested
    # testing trained w
    perClassPredictions=classifyWithW(trainingTestSamplesX, wPerceptron)
    svmClassPredictions=classifyWithW(trainingTestSamplesX, wSVM)
    paClassPredictions=classifyWithW(trainingTestSamplesX, wPA)
    print("Perceptron success rate: " + str(testWOnTestingSamples(perClassPredictions, trainingTestLablesY)) + "%")
    print("SVM success rate: " + str(testWOnTestingSamples(svmClassPredictions, trainingTestLablesY)) + "%")
    print("PA success rate: " + str(testWOnTestingSamples(paClassPredictions, trainingTestLablesY)) + "%")
    """

    # testing samples classification
    perClassPredictions = classifyWithW(testingSamples, wPerceptron)
    svmClassPredictions = classifyWithW(testingSamples, wSVM)
    paClassPredictions = classifyWithW(testingSamples, wPA)

    for i in range(len(perClassPredictions)):
        print(f"perceptron: {perClassPredictions[i]}, svm: {svmClassPredictions[i]}, pa: {paClassPredictions[i]}")


main()
