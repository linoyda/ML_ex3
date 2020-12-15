import numpy as np
import sys


def main():
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







if __name__ == "__main__":
    main()

