import numpy as np
import random
import theano.tensor as T
import theano
from Library.bayesian_neural_networks import pack_weights, unpack_weights, evaluation, normalization, init_weights


def d_sgld_RR(a_0, theta, d, M, nb_iter, nb_sampling, K, y_train, X_train, y_test, X_test, batch_size):
    """
    Implements Distributed Stochastic Gradient Langevin Dynamics for Bayesian Logistic Regression by Ahn et al.
    Favorable assumptions made:
        - Assumes response delay unchanged and same at all workers
        - Assumes trajectory length to be the same across all workers
    """
    arr_acc = np.zeros(nb_sampling)

    ''' Define the neural network here '''
    num_vars = d * n_hidden + n_hidden * 2 + 2  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances
    X = T.matrix('X')  # Feature matrix
    y = T.vector('y')  # labels

    w_1 = T.matrix('w_1')  # weights between input layer and hidden layer
    b_1 = T.vector('b_1')  # bias vector of hidden layer
    w_2 = T.vector('w_2')  # weights between hidden layer and output layer
    b_2 = T.scalar('b_2')  # bias of output

    N = T.scalar('N')  # number of observations

    log_gamma = T.scalar('log_gamma')  # variances related parameters
    log_lambda = T.scalar('log_lambda')

    prediction = T.dot(T.nnet.relu(T.dot(X, w_1) + b_1), w_2) + b_2

    ''' define the log posterior distribution '''
    log_lik_data = -0.5 * X.shape[0] * (T.log(2 * np.pi) - log_gamma) - (T.exp(log_gamma) / 2) * T.sum(
        T.power(prediction - y, 2))
    log_prior_data = (a0 - 1) * log_gamma - b0 * T.exp(log_gamma) + log_gamma
    log_prior_w = -0.5 * (num_vars - 2) * (T.log(2 * np.pi) - log_lambda) - (T.exp(log_lambda) / 2) * (
            (w_1 ** 2).sum() + (w_2 ** 2).sum() + (b_1 ** 2).sum() + b_2 ** 2)

    # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
    log_posterior = (log_lik_data * N / X.shape[0] + log_prior_data + log_prior_w)
    dw_1, db_1, dw_2, db_2, d_log_gamma = T.grad(log_posterior, [w_1, b_1, w_2, b_2, log_gamma])

    # automatic gradient
    logp_gradient = theano.function(
        inputs=[X, y, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
        outputs=[dw_1, db_1, dw_2, db_2, d_log_gamma])

    # prediction function
    nn_predict = theano.function(inputs=[X, w_1, b_1, w_2, b_2], outputs=prediction)

    for i in range(nb_sampling):
        if i == 0:
            arr_acc[i] = evaluation(M, theta, d, n_hidden, nn_predict, X_test, y_test, mean_X_train, std_X_train, mean_y_train, std_y_train)
        else:
            # assign workers to chains
            if M > K:
                assignments = np.random.permutation(K)  # assignments[i] = worker sampling chain i
                chain = i % K

                # each chain corresponds M//K of the particles
                theta[chain*(M//K) : M//K + chain*(M//K)] = sample_traj(i, a_0, theta[chain*(M//K) : M//K + chain*(M//K)],\
                    nb_iter, X_train[assignments[chain]*X_train.shape[0]//K : ((assignments[chain] + 1)*X_train.shape[0]) // K, :],\
                    y_train[assignments[chain]*X_train.shape[0]//K : ((assignments[chain] + 1)*X_train.shape[0]) // K], num_vars, batch_size, logp_gradient)
            else:
                # In this case, one particle per chain
                assignments = np.random.permutation(K)  # assignments[i] = worker sampling chain i
                chain = i % M

                theta[chain: chain + 1] = sample_traj(i, a_0, theta[chain: chain + 1], nb_iter,\
                      X_train[assignments[chain]*X_train.shape[0]//K : ((assignments[chain] + 1)*X_train.shape[0]) // K, :],\
                      y_train[assignments[chain]*X_train.shape[0]//K : ((assignments[chain] + 1)*X_train.shape[0]) // K], num_vars, batch_size, logp_gradient)

            arr_acc[i] = evaluation(M, theta, d, n_hidden, nn_predict, X_test, y_test, mean_X_train, std_X_train, mean_y_train, std_y_train)
    return arr_acc


def sample_traj(i, a_0, theta, nb_iter, X_train, y_train, num_vars, batchsize, logp_gradient):
    """
    samples a trajectory of length nb_iter corresponding to a chain starting at theta and on worker/model
    """
    N0 = X_train.shape[0]  # number of observations
    grad_theta = np.zeros([len(theta), num_vars])  # gradient with respect to particles theta
    for t in range(nb_iter):
        epsilon_t = a_0 * (0.5 + (i*nb_iter + t)) ** (-0.55)

        ''' Compute grad_theta '''
        # sub-sampling
        batch = [ii % N0 for ii in range(t * batchsize, (t + 1) * batchsize)]
        for m in range(len(theta)):
            w1, b1, w2, b2, loggamma = unpack_weights(theta[m, :], d, n_hidden)
            dw1, db1, dw2, db2, dloggamma = logp_gradient(X_train[batch, :], y_train[batch], w1, b1, w2,
                                                                      b2, loggamma, loglambda, N0)
            grad_theta[m, :] = pack_weights(dw1, db1, dw2, db2, dloggamma)

        theta = theta + (1/len(theta)) * (epsilon_t * 0.5 * grad_theta + np.random.normal(0, epsilon_t ** 2))

    return theta


if __name__ == '__main__':
    np.random.seed(0)
    ''' Parameters'''
    batch_size = 100  # batch size to draw
    n_hidden = 50  # number of units per hidden layer
    nb_iter = 200  # number of local iterations
    M = 20  # number of particles
    a0 = 1  # gamma distribution first parameter
    b0 = 0.1  # rate=1/scale of gamma distribution
    betta = 0.9
    nb_exp = 10  # number of trials
    train_ratio = 0.9  # We create the train and test sets with 90% and 10% of the data resp.
    nb_sampling = 21  # number of global iterations = I + 1 (first iteration for initialization)
    nb_devices = 2  # number of agents
    a_0 = 0.01  # DSGLD learning rate =  0.0005 for year dataset and 0.01 for all other datasets
    avg_rmse = np.zeros(nb_sampling)
    loglambda = 1

    ''' load data file '''
    data = np.loadtxt('../data/kin8nm.txt')
    # TODO: Please make sure that Last column is the label and the other columns are the features
    X_input = data[:, range(data.shape[1] - 1)]
    y_input = data[:, data.shape[1] - 1]

    for exp in range(nb_exp):
        print('Trial = ', exp + 1)

        permutation = np.arange(X_input.shape[0])
        random.shuffle(permutation)

        size_train = int(np.round(X_input.shape[0] * train_ratio))
        index_train = permutation[0: size_train]
        index_test = permutation[size_train:]

        ''' build the training and testing data set'''
        X_train, y_train = X_input[index_train, :], y_input[index_train]
        X_test, y_test = X_input[index_test, :], y_input[index_test]
        d = X_train.shape[1]

        ''' Normalize dataset '''
        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0] = 1
        mean_X_train = np.mean(X_train, 0)

        mean_y_train = np.mean(y_train)
        std_y_train = np.std(y_train)
        X_train, y_train = normalization(X_train, y_train, mean_X_train, std_X_train, mean_y_train, std_y_train)

        ''' Initialize particles'''
        num_vars = d * n_hidden + n_hidden * 2 + 2  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 1 variances
        theta = np.zeros([M, num_vars])

        for i in range(M):
            w1, b1, w2, b2, loggamma = init_weights(a0, b0, d, n_hidden)
            # use better initialization for gamma
            ridx = np.random.choice(range(X_train.shape[0]),\
                                    np.min([X_train.shape[0], 1000]), replace=False)
            A = np.dot(X_train[ridx, :], w1) + b1
            A[A < 0] = 0
            y_hat = np.dot(A, w2) + b2
            loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            theta[i, :] = pack_weights(w1, b1, w2, b2, loggamma)

        ''' Run DSVGD server with Round Robin (RR) scheduling '''
        sgld_rmse = d_sgld_RR(a_0, theta, d, M, nb_iter, nb_sampling, nb_devices, y_train, X_train, y_test, X_test, batch_size)
        avg_rmse += sgld_rmse

    print('BNN regression accuracy with DSGLD as function of comm. rounds =  ', repr(avg_rmse / nb_exp))
