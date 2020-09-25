import numpy as np
import random
import theano.tensor as T
import theano
from Library.bayesian_neural_networks_classification import pack_weights, unpack_weights, evaluation, init_weights
import pickle
"""
    Bayesian Neural network (BNN) for multi-label classification using SGLD (Welling & Teh, 2011).
    We use similar setting to Hernandez-Lobato & Adams (2015) with the difference that we use a 
    categorical distribution to model the log-likelihood p(y|W, X) parameterized by the output of the BNN.
    We fix the covariance of the prior on the weight to loglambda = 1.
"""


def sgld(a_0, theta, d, M, nb_iter, y_train, X_train, y_test, X_test, batchsize):
    """
    Implements Stochastic Gradient Langevin Dynamic for Bayesian Logistic Regression by (Welling & Teh, 2011).
    """

    ''' Define the neural network here '''
    num_vars = d * n_hidden + n_hidden + n_hidden * number_labels + number_labels  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden*10; b2 = 10;
    X = T.matrix('X')  # Feature matrix
    y = T.vector('y', dtype='int64')  # vector of labels
    log_lambda = T.scalar('log_lambda')

    w_1 = T.matrix('w_1')  # weights between input layer and hidden layer
    b_1 = T.vector('b_1')  # bias vector of hidden layer
    w_2 = T.matrix('w_2')  # weights between hidden layer and output layer
    b_2 = T.vector('b_2')  # bias of outputs

    N = T.scalar('N')  # number of observations

    prediction = T.nnet.softmax(T.dot(T.nnet.relu(T.dot(X, w_1) + b_1), w_2) + b_2)

    ''' define the log posterior distribution '''
    log_lik_data = T.sum(T.log(prediction)[T.arange(y.shape[0]), y])
    log_prior_w = -0.5 * num_vars * (T.log(2 * np.pi) - log_lambda) - (T.exp(log_lambda) / 2) * (
            (w_1 ** 2).sum() + (w_2 ** 2).sum() + (b_1 ** 2).sum() + (b_2 ** 2).sum())

    # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
    log_posterior = log_lik_data * N / X.shape[0] + log_prior_w
    dw_1, db_1, dw_2, db_2 = T.grad(log_posterior, [w_1, b_1, w_2, b_2])

    # automatic gradient
    logp_gradient = theano.function(
        inputs=[X, y, w_1, b_1, w_2, b_2, log_lambda, N],
        outputs=[dw_1, db_1, dw_2, db_2])

    # prediction function
    nn_predict = theano.function(inputs=[X, w_1, b_1, w_2, b_2], outputs=prediction)

    N0 = X_train.shape[0]  # number of observations
    grad_theta = np.zeros([M, num_vars])  # gradient

    for t in range(nb_iter):
        epsilon_t = a_0 * (0.5 + t) ** (-0.55)
        ''' Compute grad_theta '''
        # sub-sampling
        batch = [ii % N0 for ii in range(t * batchsize, (t + 1) * batchsize)]
        for m in range(M):
            w1, b1, w2, b2 = unpack_weights(theta[m, :], d, n_hidden, number_labels)
            dw1, db1, dw2, db2 = logp_gradient(X_train[batch, :], y_train[batch], w1, b1, w2, b2, loglambda, N0)
            grad_theta[m, :] = pack_weights(dw1, db1, dw2, db2)

        # update model parameters
        theta = theta + (1/M) * (epsilon_t * 0.5 * grad_theta + np.random.normal(0, epsilon_t ** 2))

    return evaluation(M, theta, d, n_hidden, number_labels, nn_predict, X_test, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    ''' Parameters '''
    alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
    nb_iter = 2000  # number of iterations= I \times L, i.e., 20,000 for 20 devices
    betta = 0.9  # for momentum update
    nb_exp = 20  # number of trials
    M = 20  # number of particles
    avg_accuracy = 0  # average accuracy after iterating nb_exp times through dataset
    avg_llh = 0  # average log-likelihood after iterating nb_exp times through dataset
    a_0 = 0.001  # initial learning rate for SGLD, fixed via validation
    batchsize = 100
    n_hidden = 100  # number of neurons in the hidden layer
    number_labels = 10  # number of labels for the MNIST dataset
    loglambda = 1  # log precision of weight prior

    ''' load data file '''
    # we use pickle for a faster import
    # TODO: please run create_pickled.py file on your dataset of choice to obtain the .pkl binary file
    with open("../data/pickled_mnist.pkl", "br") as fh:
        data = pickle.load(fh)

    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2].astype(int)
    test_labels = data[3].astype(int)

    for exp in range(nb_exp):
        print('Trial ', exp + 1)

        # shuffle the training and test set
        permutation = np.arange(train_imgs.shape[0])
        random.shuffle(permutation)
        index_train = permutation

        permutation = np.arange(test_imgs.shape[0])
        random.shuffle(permutation)
        index_test = permutation

        ''' build the training and testing data set '''
        X_train, y_train = train_imgs[index_train, :], train_labels[index_train].flatten()
        X_test, y_test = test_imgs[index_test], test_labels[index_test].flatten()

        d = X_train.shape[1]  # dimension of each particle = dimensions of a data point
        num_vars = d * n_hidden + n_hidden + n_hidden * number_labels + number_labels  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden*10; b2 = 10;

        ''' Initialize particles'''
        theta = np.zeros([M, num_vars])

        ''' Initialize particles'''
        theta = np.zeros([M, num_vars])
        for i in range(M):
            w1, b1, w2, b2 = init_weights(d, n_hidden, number_labels)
            theta[i, :] = pack_weights(w1, b1, w2, b2)

        curr_accuracy, curr_llh = sgld(a_0, theta, d, M, nb_iter, y_train, X_train, y_test, X_test, batchsize)

        avg_accuracy += curr_accuracy
        avg_llh += curr_llh

    print('BNN multilabel classification accuracy with SGLD = ', repr(avg_accuracy / nb_exp))
    print('BNN multilabel classification llh with SGLD = ', repr(avg_llh / nb_exp))

