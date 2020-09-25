import numpy as np
import random
import theano.tensor as T
import theano
from Library.bayesian_neural_networks import normalization, init_weights, pack_weights, unpack_weights, evaluation
"""
    This File implements Federated Averaging (McMahan et al., 2017) with round robin scheduling
    of one worker per global iteration
"""


def fed_sgd_client(iterpvi, theta, X_train, y_train, X_test, y_test, batch_size, nb_iter, M, n_hidden, a0, b0, alpha_ada, betta, loggamma):
    """
    samples a trajectory of length nb_iter corresponding to a chain starting at theta and on worker/model
    """
    d = X_train.shape[1]  # number of data, dimension
    num_vars = d * n_hidden + n_hidden * 2 + 1  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1;

    ''' Define the neural network here '''
    X = T.matrix('X')  # Feature matrix
    y = T.vector('y')  # labels

    w_1 = T.matrix('w_1')  # weights between input layer and hidden layer
    b_1 = T.vector('b_1')  # bias vector of hidden layer
    w_2 = T.vector('w_2')  # weights between hidden layer and output layer
    b_2 = T.scalar('b_2')  # bias of output

    N = T.scalar('N')  # number of observations

    log_gamma = T.scalar('log_gamma')  # variances related parameters

    prediction = T.dot(T.nnet.relu(T.dot(X, w_1) + b_1), w_2) + b_2

    ''' define the log likelihood distribution '''
    log_lik_data = -0.5 * X.shape[0] * (T.log(2 * np.pi) - log_gamma) - (T.exp(log_gamma) / 2) * T.sum(T.power(prediction - y, 2))

    log_ll = log_lik_data * N / X.shape[0]
    dw_1, db_1, dw_2, db_2, = T.grad(log_ll, [w_1, b_1, w_2, b_2])

    # automatic gradient
    logll_gradient = theano.function(
        inputs=[X, y, w_1, b_1, w_2, b_2, log_gamma, N],
        outputs=[dw_1, db_1, dw_2, db_2])

    # prediction function
    nn_predict = theano.function(inputs=[X, w_1, b_1, w_2, b_2], outputs=prediction)

    N0 = X_train.shape[0]  # number of observations

    grad_theta = np.zeros([1, num_vars])  # gradient
    # adagrad with momentum parameters
    fudge_factor = 1e-6
    historical_grad = 0

    if iterpvi == -1:
        return nn_predict

    for iter in range(nb_iter):

        # sub-sampling
        batch = [i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size)]
        for i in range(M):
            w1, b1, w2, b2 = unpack_weights(theta[i, :], d, n_hidden, True)
            dw1, db1, dw2, db2 = logll_gradient(X_train[batch, :], y_train[batch], w1, b1, w2, b2, loggamma, N0)
            grad_theta[i, :] = pack_weights(dw1, db1, dw2, db2, 0, True)

        # adagrad
        if iter == 0:
            historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
        else:
            historical_grad = betta * historical_grad + (1 - betta) * np.multiply(grad_theta, grad_theta)
        adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
        theta = theta + alpha_ada * adj_grad

    return theta, nn_predict


def server(X_train, y_train, X_test, y_test, batch_size, nb_iter, M, n_hidden, a0, b0, alpha_ada, betta):
    """
    Implements Federated Averaging (McMahan et al., 2017) with round robin scheduling
    of one worker per global iteration
    """

    ''' The data sets are normalized so that the input features and the targets have zero mean and unit variance '''
    std_X_train = np.std(X_train, 0)
    std_X_train[std_X_train == 0] = 1
    mean_X_train = np.mean(X_train, 0)

    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)

    # normalization
    X_train, y_train = normalization(X_train, y_train, mean_X_train, std_X_train, mean_y_train, std_y_train)

    tot_size = X_train.shape[0]  # total size of datasets
    acc = np.zeros(nb_global+1)

    # initialize weights (M = 1 in this case)
    d = X_train.shape[1]  # number of data, dimension
    num_vars = d * n_hidden + n_hidden * 2 + 1  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1;
    theta = np.zeros([1, num_vars])
    w1, b1, w2, b2 = init_weights(0, 0, d, n_hidden, True)
    # use a good initialization for gamma: note that gamma is irrelevant in a frequentist approach no matter the its value
    ridx = np.random.choice(range(X_train.shape[0]), np.min([X_train.shape[0], 1000]), replace=False)
    A1 = np.dot(X_train[ridx, :], w1) + b1
    A1[A1 < 0] = 0  # relu
    y_hat = np.dot(A1, w2) + b2
    loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
    theta[0, :] = pack_weights(w1, b1, w2, b2, 0, True)

    for i in range(-1, nb_global):

        # pick one agent in a round robin manner
        curr_agent = i % nb_devices

        # get local dataset at each agent
        X_curr, y_curr = X_train[curr_agent*X_train.shape[0]//nb_devices: ((curr_agent + 1)*X_train.shape[0]) // nb_devices, :],\
                         y_train[curr_agent*X_train.shape[0]//nb_devices : ((curr_agent + 1)*X_train.shape[0]) // nb_devices]

        if i == -1:
            nn_predict = fed_sgd_client(i, theta, X_curr, y_curr, X_test, y_test, batch_size, nb_iter, M, n_hidden, a0, b0, alpha_ada, betta, loggamma)
            acc[i + 1] = evaluation(M, theta, d, n_hidden, nn_predict, X_test, y_test, mean_X_train, std_X_train, mean_y_train, std_y_train)
        else:
            n_curr = X_curr.shape[0]
            # update global parameters after re-weighting with respect to local dataset sizes of each agent
            curr_theta, nn_predict = fed_sgd_client(i, theta, X_curr, y_curr, X_test, y_test, batch_size, nb_iter, M, n_hidden, a0, b0, alpha_ada, betta, loggamma)
            theta = (n_curr/tot_size) * curr_theta + (1-n_curr/tot_size) * theta
            acc[i+1] = evaluation(M, theta, d, n_hidden, nn_predict, X_test, y_test, mean_X_train, std_X_train, mean_y_train, std_y_train)

    return acc


if __name__ == '__main__':
    np.random.seed(0)
    ''' Parameters'''
    batch_size = 100  # batch size to draw
    n_hidden = 50  # number of units per hidden layer
    nb_iter = 200  # number of SGD iterations
    M = 1  # number of particles
    a0 = 1  # gamma distribution first parameter
    b0 = 0.1  # rate=1/scale of gamma distribution
    alpha_ada = 1e-3
    my_lambda = 0.55  # KDE bandwidth
    betta = 0.9
    nb_exp = 30  # number of trials
    train_ratio = 0.9  # We create the train and test sets with 90% and 10% of the data
    nb_global = 20  # number of global iterations = I
    nb_devices = 20  # number of agents
    avg_rmse, avg_ll = np.zeros(nb_global+1), np.zeros(nb_global+1)

    ''' load data file '''
    data = np.loadtxt('../data/kin8nm.txt')
    # TODO: Please make sure that Last column is the label and the other columns are the features
    X_input = data[:, range(data.shape[1] - 1)]
    y_input = data[:, data.shape[1] - 1]

    for exp in range(nb_exp):
        print('Trial ', exp+1)
        permutation = np.arange(X_input.shape[0])
        random.shuffle(permutation)

        size_train = int(np.round(X_input.shape[0] * train_ratio))
        index_train = permutation[0: size_train]
        index_test = permutation[size_train:]

        ''' build the training and testing data set'''
        X_train, y_train = X_input[index_train, :], y_input[index_train]
        X_test, y_test = X_input[index_test, :], y_input[index_test]

        ''' Run FedAvg server with Round Robin (RR) scheduling '''
        rmse = server(X_train, y_train, X_test, y_test, batch_size, nb_iter, M, n_hidden, a0, b0, alpha_ada, betta)
        avg_rmse += rmse

    print('BNN regression accuracy with FedAvg as function of comm. rounds = ', repr(avg_rmse / nb_exp))
