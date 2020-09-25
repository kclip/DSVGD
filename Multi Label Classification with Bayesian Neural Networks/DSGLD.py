import numpy as np
import random
import theano.tensor as T
import theano
from BNN_multiclassification.DSVGD_i import pack_weights, unpack_weights, evaluation, init_weights
import pickle


def d_sgld_RR(a_0, theta, number_labels, d, M, nb_iter, nb_global, K, y_train, X_train, y_test, X_test, batch_size):
    """
    Implements Distributed Stochastic Gradient Langevin Dynamics for Bayesian Logistic Regression by Ahn et al. 2014
    Favorable assumptions made:
        - Assumes response delay unchanged and same at all workers
        - Assumes trajectory length to be the same across all workers
    """
    arr_acc = np.zeros(nb_global)
    arr_llh = np.zeros(nb_global)

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

    for i in range(nb_global):
        if i == 0:
            # evaluate initial particles
            arr_acc[i], arr_llh[i] = evaluation(M, theta, d, n_hidden, number_labels, nn_predict, X_test, y_test)
        else:
            # assign workers to chains
            if M > K:
                assignments = np.random.permutation(K)  # assignments[i] = worker sampling chain i
                chain = i % K

                # each chain corresponds to M//K particles
                theta[chain*(M//K) : M//K + chain*(M//K)] = sample_traj(i, a_0, theta[chain*(M//K) : M//K + chain*(M//K)],\
                    nb_iter, X_train[assignments[chain]*X_train.shape[0]//K : ((assignments[chain] + 1)*X_train.shape[0]) // K, :],\
                    y_train[assignments[chain]*X_train.shape[0]//K : ((assignments[chain] + 1)*X_train.shape[0]) // K], num_vars, batch_size, logp_gradient)
            else:
                # in this case, one particle per chain
                assignments = np.random.permutation(K)  # assignments[i] = worker sampling chain i
                chain = i % M

                theta[chain: chain + 1] = sample_traj(i, a_0, theta[chain: chain + 1], nb_iter,\
                      X_train[assignments[chain]*X_train.shape[0]//K : ((assignments[chain] + 1)*X_train.shape[0]) // K, :],\
                      y_train[assignments[chain]*X_train.shape[0]//K : ((assignments[chain] + 1)*X_train.shape[0]) // K], num_vars, batch_size, logp_gradient)

            arr_acc[i], arr_llh[i] = evaluation(M, theta, d, n_hidden, number_labels, nn_predict, X_test, y_test)
    return arr_acc, arr_llh


def sample_traj(i, a_0, theta, nb_iter, X_train, y_train, num_vars, batchsize, logp_gradient):
    """
    samples a trajectory of length nb_iter corresponding to a chain starting at theta and on worker/model
    """
    N0 = X_train.shape[0]  # number of observations
    grad_theta = np.zeros([len(theta), num_vars])  # gradient
    for t in range(nb_iter):
        epsilon_t = a_0 * (0.5 + (i*nb_iter + t)) ** (-0.55)

        ''' Compute grad_theta '''
        # sub-sampling
        batch = [ii % N0 for ii in range(t * batchsize, (t + 1) * batchsize)]
        for m in range(len(theta)):
            w1, b1, w2, b2 = unpack_weights(theta[m, :], d, n_hidden, number_labels)
            dw1, db1, dw2, db2 = logp_gradient(X_train[batch, :], y_train[batch], w1, b1, w2, b2, loglambda, N0)
            grad_theta[m, :] = pack_weights(dw1, db1, dw2, db2)

        # particles update
        theta = theta + (1/len(theta)) * (epsilon_t * 0.5 * grad_theta + np.random.normal(0, epsilon_t ** 2))

    return theta


if __name__ == '__main__':
    np.random.seed(0)
    ''' Parameters'''
    batch_size = 100  # batch size to draw
    n_hidden = 100  # number of neurons of the hidden layer
    nb_iter = 200  # number of local iterations
    M = 20  # number of particles
    betta = 0.9  # for momentum update
    nb_exp = 10  # number of trials
    nb_global = 11  # number of global iterations + 1 = I + 1
    nb_devices = 2  # number of agents
    a_0 = 0.0005  # for DSGLD learning rate numerator
    number_labels = 10  # number of labels in MNIST dataset
    loglambda = 1  # log precision of weight prior
    arr_avg_acc, arr_avg_llh = np.zeros(nb_global), np.zeros(nb_global)

    ''' load data file '''
    # we use pickle for a faster import
    # TODO: please run create_pickled.py file on your dataset of choice to obtain the .pkl binary file then sepcify its directory below
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

        '''Run DSGLD server with Round Robin (RR) scheduling'''
        dsgld_acc, dsgld_llh = d_sgld_RR(a_0, theta, number_labels, d, M, nb_iter, nb_global, nb_devices, y_train, X_train, y_test, X_test, batch_size)
        arr_avg_acc += dsgld_acc
        arr_avg_llh += dsgld_llh

    print('BNN multilabel classification accuracy with DSGLD as function of comm. rounds = ', repr(arr_avg_acc / nb_exp))
    print('BNN multilabel classification llh with DSGLD as function of comm. rounds = ', repr(arr_avg_llh / nb_exp))
