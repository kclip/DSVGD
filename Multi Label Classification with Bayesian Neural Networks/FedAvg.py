import numpy as np
import random
import theano.tensor as T
import theano
import pickle
from Library.bayesian_neural_networks_classification import init_weights, pack_weights, unpack_weights, evaluation_Fed
"""
    Bayesian Neural network (BNN) for multi-label classification using FedAvg (McMahan et al., 2017).
    We use similar setting to Hernandez-Lobato & Adams (2015) with the difference that we use a 
    categorical distribution to model the log-likelihood p(y|W, X) parameterized by the output of the BNN.
    We fix the covariance of the prior on the weight to loglambda = 1.
    We schedule one agent/client at a time for fairness with DSVGD.
"""


def fed_avg_client(iterpvi, num_vars, theta, X_train, y_train, batch_size, nb_iter, M, n_hidden,
                   alpha_ada, betta):
    d = X_train.shape[1]  # number of dimensions

    ''' Define the neural network here '''
    X = T.matrix('X')  # Feature matrix
    y = T.vector('y', dtype='int64')  # vector of labels

    w_1 = T.matrix('w_1')  # weights between input layer and hidden layer
    b_1 = T.vector('b_1')  # bias vector of hidden layer
    w_2 = T.matrix('w_2')  # weights between hidden layer and output layer
    b_2 = T.vector('b_2')  # bias of outputs

    N = T.scalar('N')  # number of observations

    prediction = T.nnet.softmax(T.dot(T.nnet.relu(T.dot(X, w_1) + b_1), w_2) + b_2)

    ''' define the log likelihood distribution '''
    log_lik_data = T.sum(T.log(prediction)[T.arange(y.shape[0]), y])

    log_ll = log_lik_data * N / X.shape[0]
    dw_1, db_1, dw_2, db_2, = T.grad(log_ll, [w_1, b_1, w_2, b_2])

    # automatic gradient
    logll_gradient = theano.function(
        inputs=[X, y, w_1, b_1, w_2, b_2, N],
        outputs=[dw_1, db_1, dw_2, db_2])

    # prediction function
    nn_predict = theano.function(inputs=[X, w_1, b_1, w_2, b_2], outputs=prediction)

    N0 = X_train.shape[0]  # number of observations

    grad_theta = np.zeros([1, num_vars])  # gradient

    # adagrad with momentum parameters
    fudge_factor = 1e-6
    historical_grad = 0

    if iterpvi == -1:
        # initialization, to evaluate initial random model performance
        return nn_predict

    for iter in range(nb_iter):

        # sub-sampling
        batch = [i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size)]
        for i in range(M):
            w1, b1, w2, b2 = unpack_weights(theta[i, :], d, n_hidden, number_labels)
            dw1, db1, dw2, db2 = logll_gradient(X_train[batch, :], y_train[batch], w1, b1, w2, b2, N0)
            grad_theta[i, :] = pack_weights(dw1, db1, dw2, db2)

        # adagrad
        if iter == 0:
            historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
        else:
            historical_grad = betta * historical_grad + (1 - betta) * np.multiply(grad_theta, grad_theta)
        adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))

        # update model parameters
        theta = theta + alpha_ada * adj_grad

    return theta, nn_predict


def server(C, X_train, y_train, X_test, y_test, batch_size, nb_iter, M, n_hidden, alpha_ada, betta, number_labels):
    """
    Implements Federated Averaging (McMahan et al., 2017) with round robin scheduling
    of one worker/agent per global iteration (i.e., comm. round)
    """

    tot_size = X_train.shape[0]  # total size of datasets, i.e, number of training images
    acc = np.zeros(nb_global + 1)
    llh = np.zeros(nb_global + 1)

    # initialize weights (M = 1 particle in this case)
    d = X_train.shape[1]  # number of data dimensions
    num_vars = d * n_hidden + n_hidden + n_hidden * number_labels + number_labels  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden*10; b2 = 10;
    theta = np.zeros([1, num_vars])  # matrix of parameters
    w1, b1, w2, b2 = init_weights(d, n_hidden, number_labels)

    theta[0, :] = pack_weights(w1, b1, w2, b2)

    for i in range(-1, nb_global):
        # scheduled agents numbers (only one agent in our case)
        scheduled_agents = np.random.choice(nb_devices, int(nb_devices * C), replace=False)
        all_theta = np.zeros((nb_devices, theta.shape[1]))  # all models to be received from all scheduled agents
        scheduled_size = np.zeros((nb_devices, theta.shape[1]))

        if i == -1:
            # initialization: to evaluate random parameters performance
            k = 0
            X_curr, y_curr = X_train[k * X_train.shape[0] // nb_devices: ((k + 1) * X_train.shape[0]) // nb_devices, :], \
                             y_train[k * X_train.shape[0] // nb_devices: ((k + 1) * X_train.shape[0]) // nb_devices]
            nn_predict = fed_avg_client(i, num_vars, theta, X_curr, y_curr, batch_size, nb_iter, M,
                                        n_hidden, alpha_ada, betta)
        else:
            for k in scheduled_agents:
                # get scheduled agent local dataset
                X_curr, y_curr = X_train[k * X_train.shape[0] // nb_devices: ((k + 1) * X_train.shape[0]) // nb_devices, :],\
                                 y_train[k * X_train.shape[0] // nb_devices: ((k + 1) * X_train.shape[0]) // nb_devices]
                n_curr = X_curr.shape[0]  # scheduled agent dataset size
                scheduled_size[k] = n_curr

                # get new parameters updated by the k-th scheduled agent
                all_theta[k], nn_predict = fed_avg_client(i, num_vars, theta, X_curr, y_curr,
                                                          batch_size, nb_iter, M, n_hidden, alpha_ada, betta)

            new_theta = np.zeros((1, theta.shape[1]))  # new server parameters
            for k in range(nb_devices):
                if k in scheduled_agents:
                    # if agent k was scheduled, get its new model
                    new_theta += scheduled_size[k] / tot_size * all_theta[k]
                else:
                    X_curr = X_train[k * X_train.shape[0] // nb_devices: ((k + 1) * X_train.shape[0]) // nb_devices, :]
                    n_curr = X_curr.shape[0]
                    new_theta += n_curr / tot_size * theta

            theta = new_theta

        acc[i + 1], llh[i + 1] = evaluation_Fed(theta, d, n_hidden, nn_predict, X_test, y_test, number_labels)

    return acc, llh


if __name__ == '__main__':
    np.random.seed(0)
    ''' Parameters'''
    nb_iter = 200  # number of local SGD iterations
    M = 1  # number of particles
    number_labels = 10  # number of labels in MNIST
    alpha_ada = 1e-3  # constant rate coeff. used in AdaGrad
    betta = 0.9  # for momentum update
    nb_exp = 10  # number of random trials
    nb_global = 10  # number of global iterations = I = number of comm. rounds
    nb_devices = 2  # number of agents
    avg_rmse, avg_ll = np.zeros(nb_global + 1), np.zeros(nb_global + 1)
    C = 1/nb_devices  # proportion of agents to schedule
    batch_size = 100  # batch size to draw
    n_hidden = 100  # number of neurons of the hidden layer
    array_acc = np.zeros(nb_global + 1)
    array_llh = np.zeros(nb_global + 1)

    ''' load Fashion MNIST data file '''
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

        ''' Training Bayesian neural network with FedAvg '''
        curr_acc, curr_llh = server(C, X_train, y_train, X_test, y_test, batch_size, nb_iter, M, n_hidden,
                                    alpha_ada, betta, number_labels)
        array_acc += curr_acc
        array_llh += curr_llh

    print('BNN multilabel classification accuracy with FedAvg as function of comm. rounds = ', repr(array_acc / nb_exp))
    print('BNN multilabel classification llh with FedAVg as function of comm. rounds = ', repr(array_llh / nb_exp))
