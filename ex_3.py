import random
import numpy as np
import sys


def main():
    total_epoches = 100
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
    taken_sorted = random.sample(range(1, 55001), 5000)
    taken_sorted.sort()

    with open('train_x_short', 'w') as out_file_1:
        i = 1
        train_x_open = open(train_x)
        for curr_line_1 in train_x_open:
            if i in taken_sorted:
                out_file_1.write(curr_line_1)
            i = i + 1
        train_x_open.close()
    with open('train_y_short', 'w') as out_file_2:
        i = 1
        train_y_open = open(train_y)
        for curr_line_2 in train_y_open:
            if i in taken_sorted:
                out_file_2.write(curr_line_2)
            i = i + 1
        train_y_open.close()

    # Now we'll initialize the random parameters.
    # The first number - the amount of neurons in incoming layer.
    # The second number - the amount of neurons in outcoming layer.

    w1 = np.random.randn(300, 784) * np.sqrt(2 / 784)
    b1 = np.full((300, 1), 0)
    w2 = np.random.randn(10, 300) * np.sqrt(2 / 300)
    b2 = np.full((10, 1), 0)
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    # We got two files - train_x_short, train_y_short - (each contains 5000 examples) those will be our training set.
    train_x_short = np.loadtxt("train_x_short")
    train_y_short = np.loadtxt("train_y_short")

    for curr_epoch in range(total_epoches):

        # Shuffle the examples to avoid overfitting
        shuffle_list = list(zip(train_x_short, train_y_short))
        np.random.shuffle(shuffle_list)

        for x, y in zip(train_x_short, train_y_short):
            w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]

            z1 = np.dot(w1, x) + b1
            h1 = sigmoid(z1)
            z2 = np.dot(w2, h1) + b2
            h2 = softmax(z2)


# sigmoid = lambda x: 1 / (1 + np.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fprop(x, y, params):
    # Follows procedure given in notes
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)
    # loss = -(y * np.log(h2) + (1-y) * np.log(1-h2))
    loss = calc_negative_log_likelihood(w1, x, b1)
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    dz2 = (h2 - y)                                #  dL/dz2
    dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2
    db2 = dz2                                     #  dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['W2'].T, (h2 - y)) * sigmoid(z1) * (1-sigmoid(z1))   #  dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def calc_negative_log_likelihood(w_i, x_t, b_i):
    res = np.dot(w_i, x_t) + b_i
    return softmax(res)


def softmax(x):  # x is a 1 dimension vector.
    x = x - np.max(x)
    return np.exp(x) / np.exp(x).sum()


if __name__ == "__main__":
    main()
