import numpy as np
import sys


def main():
    total_epoches = 25
    learning_rate = 0.01
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]

    """
    Pseudo Code - Training Process:
    1. loop over the training set #EPOCH number of times.
        a. shuffle the examples.
        b. loop over the training set:
            I. pick example i
            II. forward the input instance through the network (f-prop?)
            III. calculate the loss
            IV. compute the gradients w.r.t. all the parameters (back-propagation) (b-prop)
            V. update the parameters using GD / SGD.
    """

    # Load to np-array and normalize train_x, test_x
    train_x_norm = np.loadtxt(train_x) / 255
    train_y = np.loadtxt(train_y)
    test_x_norm = np.loadtxt(test_x) / 255
    # print("first\n")

    # Now we'll initialize the random parameters.
    # The first number - the amount of neurons in incoming layer.
    # The second number - the amount of neurons in outcoming layer.

    w1 = np.random.randn(300, 784) * np.sqrt(2 / 784)
    b1 = np.full((300, 1), 0)
    w2 = np.random.randn(10, 300) * np.sqrt(2 / 300)
    b2 = np.full((10, 1), 0)
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    for curr_epoch in range(total_epoches):
        # print("curr epoch " + str(curr_epoch))
        # Shuffle the examples to avoid overfitting
        shuffle_list = list(zip(train_x_norm, train_y))
        np.random.shuffle(shuffle_list)

        for x, y in zip(train_x_norm, train_y):
            convert_to_ndarray = np.ndarray(shape=(784, 1), buffer=x)
            fprop_cache = fprop(convert_to_ndarray, y, parameters)
            bprop_cache = bprop(fprop_cache)
            update_parameters_according_to_bprop_and_eta(learning_rate, parameters, bprop_cache)

    # The training is complete. Now we'll use the trained algorithm to predict y_hat (for the given test set)
    open_test_file = open("test_y", "a")
    for x in test_x_norm:
        convert_to_ndarray_test = np.ndarray(shape=(784, 1), buffer=x)
        z1 = np.dot(parameters['w1'], convert_to_ndarray_test) + parameters['b1']
        h1 = sigmoid(z1)
        z2 = np.dot(parameters['w2'], h1) + parameters['b2']
        h2 = softmax(z2)

        # Get y_hat_prediction and write it to output file accordingly
        y_hat_prediction = str(np.argmax(h2))
        open_test_file.write(f"{y_hat_prediction}\n")
    open_test_file.close()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fprop(x, y, params):
    # Follows procedure given in notes
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)

    # calculate the loss according to y and h2
    loss = calc_negative_log_likelihood(y, h2)

    ret_cache = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret_cache[key] = params[key]
    return ret_cache


def bprop(fprop_cache):
    # Follows procedure given in class notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]

    vec = np.zeros((10, 1))
    vec[int(y)] = 1

    # This is multi-class n.n. so y is treated as a vector.
    dz2 = (h2 - vec)

    dw2 = np.dot(dz2, h1.T)
    db2 = dz2

    sig_z1 = sigmoid(z1)
    dz1 = np.dot(fprop_cache['w2'].T, (h2 - vec)) * sig_z1 * (1 - sig_z1)
    dw1 = np.dot(dz1, x.T)
    db1 = dz1
    return {'b1': db1, 'w1': dw1, 'b2': db2, 'w2': dw2}


def calc_negative_log_likelihood(y, h2):
    # This is a multi-class neuron network, so the loss is calculated with a vector.
    vec = np.zeros((10, 1))
    vec[int(y)] = 1
    res = np.sum(-vec * np.log(h2))
    return res


def softmax(x):  # x is a 1 dimension vector.
    x = x - np.max(x)
    return np.exp(x) / np.exp(x).sum()


def update_parameters_according_to_bprop_and_eta(learning_rate, parameters, bprop_cache):
    parameters['w1'] = parameters['w1'] - learning_rate * bprop_cache['w1']
    parameters['w2'] = parameters['w2'] - learning_rate * bprop_cache['w2']

    parameters['b1'] = parameters['b1'] - learning_rate * bprop_cache['b1']
    parameters['b2'] = parameters['b2'] - learning_rate * bprop_cache['b2']


if __name__ == "__main__":
    main()
