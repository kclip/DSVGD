import numpy as np
import matplotlib.pyplot as plt
import pickle
"""
    This file creates a binary file for MNIST or Fashion MNIST datasets for faster dataloading.
    Please make sure to download either datasets from : http://yann.lecun.com/exdb/mnist/
    or https://github.com/zalandoresearch/fashion-mnist
    Code is used from https://www.python-course.eu/neural_network_mnist.php
"""


def main():

    # Todo: Make sure that first column of your dataset is the label of the digit
    data_path = "../../my_code/data/MNIST/"  # Todo: Specify folder where mnist_train.csv and mnist_test.csv are located
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

    # normalize pixel values in range [0.01, 1]
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    # uncomment to write binary files for fast loading of data
    with open("../data/pickled_mnist.pkl", "bw") as fh:
        data = (train_imgs,
                test_imgs,
                train_labels,
                test_labels)
        pickle.dump(data, fh)

    # uncomment to read binary files
    # with open("pickled_mnist.pkl", "br") as fh:
    #     data = pickle.load(fh)

    # read train and test images and train and test labels
    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]

    print(test_labels[1])
    print(test_imgs[0])

    plt.figure()
    plt.imshow(test_imgs[np.random.randint(len(test_imgs))].reshape((28, 28)), cmap="Greys")
    plt.show()


if __name__ == '__main__':
    main()

