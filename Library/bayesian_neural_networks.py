import numpy as np
'''
    This file contains useful functions for the Bayesian Neural Networks regression experiment,
    most functions have been borrowed and adapted from SVGD original paper code available at:
    https://github.com/DartML/Stein-Variational-Gradient-Descent
'''


def normalization(X, y, mean_X_train, std_X_train, mean_y_train, std_y_train):
    """
    Function to normalize data
    """
    X = (X - np.full(X.shape, mean_X_train)) / \
        np.full(X.shape, std_X_train)

    if y is not None:
        y = (y - mean_y_train) / std_y_train
        return (X, y)
    else:
        return X


def init_weights(a0, b0, d, n_hidden, Freq=False):
    """
    Function to initialize weights, i.e., particles following the model in:
    Jose Miguel Hernandez-Lobato et al. Probabilistic backpropagation for scalable learning of bayesian neural networks.
    """
    w1 = 1.0 / np.sqrt(d + 1) * np.random.randn(d, n_hidden)
    b1 = np.zeros((n_hidden,))
    w2 = 1.0 / np.sqrt(n_hidden + 1) * np.random.randn(n_hidden)
    b2 = 0.
    if Freq:
        # if using a frequentist approach, i.e., FedAvg, no covariance
        return (w1, b1, w2, b2)
    else:
        loggamma = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, loggamma)


def pack_weights(w1, b1, w2, b2, loggamma, Freq=False):
    """
    Function to pack all variables into one matrix
    """
    if Freq:
        params = np.concatenate([w1.flatten(), b1, w2, [b2]])
    else:
        params = np.concatenate([w1.flatten(), b1, w2, [b2], [loggamma]])
    return params


def unpack_weights(z, d, n_hidden, Freq=False):
    """
    Function to extract all weights and biases bfor input-hidden and hidden-output connection
    in addition to covariances
    """

    w = z
    w1 = np.reshape(w[:d * n_hidden], [d, n_hidden])
    b1 = w[d * n_hidden:(d + 1) * n_hidden]

    w = w[(d + 1) * n_hidden:]

    if Freq:
        # no covariance for frequentist approaches
        w2, b2 = w[:n_hidden], w[-1]
        return (w1, b1, w2, b2)
    else:
        w2, b2 = w[:n_hidden], w[-2]
        # the last two parameters are log variance
        loggamma = w[-1]
        return (w1, b1, w2, b2, loggamma)


def evaluation(M, theta, d, n_hidden, nn_predict, X_test, y_test, mean_X_train, std_X_train, mean_y_train, std_y_train):

    # normalization
    X_test = normalization(X_test, None, mean_X_train, std_X_train, mean_y_train, std_y_train)

    # average over the output
    pred_y_test = np.zeros([M, len(y_test)])

    # Since we have M particles, we use a Bayesian view to calculate RMSE and log-likelihood '''
    for i in range(M):
        w1, b1, w2, b2, loggamma = unpack_weights(theta[i, :], d, n_hidden)
        pred_y_test[i, :] = nn_predict(X_test, w1, b1, w2, b2) * std_y_train + mean_y_train

    pred = np.mean(pred_y_test, axis=0)

    # evaluation
    svgd_rmse = np.sqrt(np.mean((pred - y_test) ** 2))

    return svgd_rmse