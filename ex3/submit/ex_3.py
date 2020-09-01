# Elad Israel 313448888

import sys
import numpy as np


# go through each column(axis=0) and zscore it, returns the updated martix
def ZScoreNormalization(matrix):
    return (matrix - matrix.mean()) / matrix.std()


# reLU activation function
def reLU(z):
    return np.maximum(0, z)


# Calculate the derivative of ReLU
def reLU_derivative(z):
    # if (z > 0)-> 1.0, else (z <= 0) ->0. same as: (z > 0) * 1.0 + (z <= 0) * 0. same as (z > 0) * 1.0
    return (z > 0) * 1.0


# softmax normalization
def softmax_norm(z):
    numerator = np.exp(z - np.max(z))
    denominator = numerator.sum()
    return numerator / denominator


"""
 forward-propagation- forward x through the network
 it's called that way, because the calculation flow is going in the natural forward direction
 from the input -> through the neural network -> to the output.
 We start from the input we have, we pass them through the network layer and calculate the actual output of the model.
"""
def forward_prop(x, parameters):
    w1, b1, w2, b2 = [parameters[key] for key in ('w1', 'b1', 'w2', 'b2')]
    # first hidden layer. h1 will contains the hidden layer's neurons
    z1 = np.dot(w1, x) + b1
    # we have to normalize the data to avoid overflow- leading to "NaN" results.
    z1 = ZScoreNormalization(z1)
    # do activation function on each of the hidden layers(we have 1)
    h1 = reLU(z1)
    # last layer is the output
    z2 = np.dot(w2, h1) + b2
    y_hat = softmax_norm(z2)
    ret = {'z1': z1, 'h1': h1, 'z2': z2, 'y_hat': y_hat}
    # add parameters array to ret dictionary
    for key in parameters:
        ret[key] = parameters[key]
    return ret


"""
We have the starting point of errors, which is the loss function, and we know how to calculate its derivative,
 and if we know how to calculate the derivative of each function from the composition, 
 we can propagate back the error from the end to the start.
 forward_params: 'z1','h1', 'z2', 'y_hat', 'w1', 'b1', 'w2', 'b2'
"""
def back_prop(x, y, forward_params, eta):
    forward_params['y_hat'][int(y)] -= 1
    loss2 = forward_params['y_hat'] - int(y)
    gradient2 = loss2
    gradient_w2 = np.dot(np.reshape(gradient2, (-1, 1)), np.reshape(forward_params['h1'], (1, -1)))
    gradient_b2 = gradient2

    # update each parameter according to the gradients of the parameter and eta
    # subtract the learning rate * gradient of the parameter from each parameter
    forward_params['w2'] -= eta * gradient_w2
    forward_params['b2'] -= eta * gradient_b2

    loss1 = np.dot(forward_params['w2'].transpose(), gradient2)

    gradient1 = np.dot(reLU_derivative(forward_params['z1']), loss1)

    gradient_w1 = np.dot(gradient1, x.transpose())
    gradient_b1 = gradient1

    # update each parameter according to the gradients of the parameter and eta
    # subtract the learning rate * gradient of the parameter from each parameter
    forward_params['w1'] -= eta * gradient_w1
    forward_params['b1'] -= eta * gradient_b1
    return forward_params


# train the neural network
def train_neural_net(training_samples_x, training_labels_y, parameters, eta, epocs):
    for i in range(epocs):
        # shuffle, seed maked the random values the same on every run
        indices = np.arange(training_samples_x.shape[0])
        np.random.seed(1)
        np.random.shuffle(indices)
        training_samples_x = training_samples_x[indices]
        training_labels_y = training_labels_y[indices]
        # no need to calculate loss because backProp works without it.
        # commented in order to be consistent with the algorithm learned in class.
        # loss_sum = 0
        for x, y in zip(training_samples_x, training_labels_y):

            # run training_x[i] through the network. h2 in forward_ret is y_hat
            forward_ret = forward_prop(x, parameters)
            # calculate loss. y_hat is the prediction. loss is a performance metric on how well the NN manages to reach
            # its goal of generating outputs as close as possible to the desired values.
            # loss_sum += -np.log(forward_ret['y_hat'][np.int(y)])
            # compute the gradients of each parameter in forward_ret
            backword_ret = back_prop(x, y, forward_ret, eta)
            for key in parameters:
                # e.g: w1=w1-eta*gradient of w1, b1=b1-eta*gradient of b1. same for w2 and b2.
                parameters[key] = backword_ret[key]
        # print(loss_sum)
    return parameters


# classify the samples using the provided W
def classify_with_parameters(testing_samples_x, trained_parameters):
    classPredictions = []
    for x in testing_samples_x:
        # forward x (row). forward returns y_hat among the parameters.
        # the biggest result in y_hat vector is x's predicted classification.
        prediction = np.argmax(forward_prop(x, trained_parameters)['y_hat'])
        classPredictions.append(prediction)
    return classPredictions


# Test W on the testing samples and return success percentages
def testWOnTestingSamples(predictions, training_test_lables_y):
    # number of rows(number of testing samples)
    testingSamplesNum = len(training_test_lables_y)
    if testingSamplesNum == 0:
        return -1
    fails = 0
    for y, y_hat in zip(training_test_lables_y, predictions):
        # wrong prediction
        if y != y_hat:
            fails += 1
    # return success percentage
    return (1 - (float(fails) / testingSamplesNum)) * 100

# writes the test predictions to a file
def write_test_predictions(predictions):
    test_y_file = open("test_y", "w")
    for prediction in predictions:
        test_y_file.write(np.str(prediction) + "\n")
    test_y_file.close()


def main():
    if len(sys.argv) < 4:
        print("Error")
        exit(-1)
    training_x = np.loadtxt(sys.argv[1], delimiter=' ', dtype="float")
    training_y = np.loadtxt(sys.argv[2], delimiter=' ', dtype="float")

    max_pixel_value = 255
    # we have to normalize the data to avoid overflow- leading to "nan" results.
    training_x = np.divide(training_x, max_pixel_value)
    testing_samples = np.loadtxt(sys.argv[3], delimiter=' ', dtype="float")

    # number of rows(number of samples)
    samplesNum = len(training_x)
    # split to 80% training set and 20% testing set
    training_samples_x, training_test_samples_x = np.split(training_x, [int(samplesNum * 0.75)])
    training_labels_y, training_test_lables_y = np.split(training_y, [int(samplesNum * 0.75)])

    # num of pixels in each example- each picture is 28*28 pixels
    input_layer_neuron_count = 784
    # number of classes
    output_layer_neuron_count = 10
    # hyper parameters- tweaked
    hidden_layer_neuron_count = 250
    eta = 0.003
    epocs = 15

    """ w1 is the weights between the input and the hidden layer.
    each line in w1 is the weights a single input neuron, to each of the hidden layer's neurons.
     initialize w1 with random values.

    """
    w1 = np.random.uniform(-0.8, 0.8, (hidden_layer_neuron_count, input_layer_neuron_count))
    b1 = np.random.uniform(-0.8, 0.8, (hidden_layer_neuron_count))
    w2 = np.random.uniform(-0.8, 0.8, (output_layer_neuron_count, hidden_layer_neuron_count))
    b2 = np.random.uniform(-0.8, 0.8, (output_layer_neuron_count))
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    trained_parameters = train_neural_net(training_samples_x, training_labels_y, parameters, eta, epocs)
    # commented because we already tested the trained net.
    # predictions = classify_with_parameters(training_test_samples_x, trained_parameters)
    # print(str(round(testWOnTestingSamples(predictions, training_test_lables_y), 2)) + "%")
    predictions = classify_with_parameters(testing_samples, trained_parameters)
    write_test_predictions(predictions)


main()
