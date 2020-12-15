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
    w1 = np.random.rand(2, 2)
    b1 = np.random.rand(2, 1)
    w2 = np.random.rand(1, 2)
    b2 = np.random.rand(1, 1)

    # We got two files - train_x_short, train_y_short - (each contains 5000 examples) those will be our training set.
    train_x_short = np.loadtxt("train_x_short")
    train_y_short = np.loadtxt("train_y_short")

    for curr_epoch in range(total_epoches):

        # Shuffle the examples.
        shuffle_list = list(zip(train_x_short, train_y_short))
        np.random.shuffle(shuffle_list)

        for x, y in zip(train_x_short, train_y_short):












if __name__ == "__main__":
    main()

