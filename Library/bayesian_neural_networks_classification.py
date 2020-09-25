import numpy as np
'''
    This file contains useful functions for the Bayesian Neural Networks classification experiment,
    most functions have been borrowed and adapted from SVGD original paper code available at:
    https://github.com/DartML/Stein-Variational-Gradient-Descent
    The only difference with bayesian_neural_networks file is the the absence of loggamma, 
    the log covariance of p(y|W, X) because it is modeled as a categorical distribution
'''


def init_weights(d, n_hidden, number_labels):
    w1 = 1.0 / np.sqrt(d + 1) * np.random.randn(d, n_hidden)
    b1 = np.zeros((n_hidden,))

    w2 = 1.0 / np.sqrt(n_hidden + 1) * np.random.randn(n_hidden, number_labels)
    b2 = np.zeros((number_labels,))
    return (w1, b1, w2, b2)


def pack_weights(w1, b1, w2, b2):
    params = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
    return params


def unpack_weights(z, d, n_hidden, number_labels):
    w = z
    w1 = np.reshape(w[:d * n_hidden], [d, n_hidden])
    b1 = w[d * n_hidden:(d + 1) * n_hidden]

    w = w[(d + 1) * n_hidden:]
    w2, b2 = np.reshape(w[:n_hidden * number_labels], [n_hidden, number_labels]), w[n_hidden * number_labels:]

    return (w1, b1, w2, b2)


def evaluation(M, theta, d, n_hidden, number_labels, nn_predict, X_test, y_test, get_prob=None):

    """
    Returns accuracy and log-likelihood for particles based methods.
    Set get_pred \neq None to get confidence.
    """
    acc = np.zeros(M)
    llh = np.zeros(M)

    '''
        Since we have M particles, we use a Bayesian view to calculate accuracy and log-likelihood
        where we average over the M particles
    '''
    for i in range(M):
        w1, b1, w2, b2, = unpack_weights(theta[i, :], d, n_hidden, number_labels)
        pred = nn_predict(X_test, w1, b1, w2, b2)
        pred_y_test = np.argmax(pred, axis=1)
        prob = np.max(pred, axis=1)

        acc[i] = len(np.where(pred_y_test == y_test)[0]) / len(y_test)
        llh[i] = np.mean(np.log(prob))

    if get_prob is None:
        return (np.mean(acc), np.mean(llh))
    else:
        # get predictive distribution (confidence) by averaging over particles
        return np.mean(pred_y_test, axis=0)


def evaluation_Fed(theta, d, n_hidden, nn_predict, X_test, y_test, number_labels, get_pred=None):
    """
    Returns accuracy and log-likelihood for FedAvg.
    Set get_pred \neq None to get confidence
    """
    prob = np.zeros([1, len(y_test)])

    w1, b1, w2, b2 = unpack_weights(theta[0, :], d, n_hidden, number_labels)

    # get output of the Neural Network
    pred = nn_predict(X_test, w1, b1, w2, b2)

    # get value of the predicted label having highest probability and its probability
    pred_y_test = np.argmax(pred, axis=1)

    prob[0, :] = np.max(pred, axis=1)

    acc = len(np.where(pred_y_test == y_test)[0]) / len(y_test)
    llh = np.mean(np.log(prob[0, :]))

    if get_pred is None:
        return acc, llh
    else:
        return prob[0, :]


def compute_reliability(M, theta, d, nb_bins, X_test, y_test, nn_predict, n_hidden, number_labels):
    """
    Return: accuracy per confidence interval given particles theta
    """
    bins = [[0] for i in range(nb_bins)]  # each bin contains indices for which prediction confidence is between ( m/M , (m+1)/M]
    acc_per_bin = [0] * nb_bins
    importance_bins = [0] * nb_bins

    # compute confidence for each datapoint averaged over all particles
    pred_conf, pred = evaluation(M, theta, d, n_hidden, number_labels, nn_predict, X_test, y_test, get_prob=True)

    for m in range(0, nb_bins):
        bins[m] = np.where((pred_conf > m / nb_bins) * (pred_conf <= (m + 1) / nb_bins))

    bin_sizes = []
    # compute accuracy per bin
    m = 0
    n = len(y_test)
    for curr_bin in bins:
        bin_sizes.append(len(curr_bin[0]))
        correct_samples_in_bin = len(np.where(y_test[curr_bin[0]] - pred[curr_bin[0]] == 0)[0])
        if len(curr_bin[0]) != 0:
            acc_per_bin[m] = correct_samples_in_bin / len(curr_bin[0])
        else:
            acc_per_bin[m] = 0
        importance_bins[m] = len(curr_bin[0]) / n

        m += 1

    acc_per_bin = np.array(acc_per_bin)
    return acc_per_bin