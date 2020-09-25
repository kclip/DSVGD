import torch
import numpy as np
import random
import theano.tensor as T
import theano
from Library.general_functions import pairwise_distances, kde, svgd_kernel
from Library.bayesian_neural_networks import pack_weights, unpack_weights, evaluation, init_weights, normalization

"""
    Bayesian Neural network Regression experiment using the setting in 
    Jose Miguel Hernandez-Lobato et al. Probabilistic backpropagation for scalable learning of bayesian neural networks.
    Please see Sec. 2 of the paper for more details.
    We borrow some of the code from SVGD original paper available at: https://github.com/DartML/Stein-Variational-Gradient-Descent
    DSVGD can be easily obtained by running SVGD twice at each scheduled agent and suitable adjusting the target distribution
    for each SVGD loop.  
    We fix the covariance of the prior on the weight to loglambda = 1.
"""


def agent_dsvgd(i, d, n_hidden, M, nb_svgd, nb_svgd_2, y_train, X_train, y_test, X_test, global_particles, local_particles, batchsize):

    sum_squared_grad = torch.zeros([M, num_vars])  # sum of gradient squared to use in Ada grad learning rate for first SVGD loop
    sum_squared_grad_2 = torch.zeros([M, num_vars])  # sum of gradient squared to use in Ada grad learning rate for second SVGD loop

    ''' Define the neural network here '''
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

    if i == 0:
        # It's the first iteration for each agent, q^(i-1) = q^(0) when i = 0
        particles = global_particles.detach().clone()
        particles.requires_grad = False
        particles_2 = local_particles.detach().clone()
        particles_2.requires_grad = False
    else:
        # download particles from previous PVI iteration
        particles = global_particles.detach().clone()
        particles.requires_grad = True

        N0 = X_train.shape[0]  # number of observations
        grad_theta = np.zeros([M, num_vars])  # gradient

        ''' First SVGD loop: update of global particles'''
        for t in range(0, nb_svgd):
            # calculating the kernel matrix and its derivative
            kxy, dxkxy = svgd_kernel(particles.detach().clone().numpy(), h=-1)

            # compute t_prev = t^(i-1)
            distance_M_j = pairwise_distances(M, particles.transpose(0, 1), local_particles.transpose(0, 1))
            t_prev = kde(M, num_vars, my_lambda, distance_M_j, 'gaussian')

            # compute qi_1 = q^(i-1)
            distance_M_i_1 = pairwise_distances(M, particles.transpose(0, 1), global_particles.transpose(0, 1))
            qi_1 = kde(M, num_vars, my_lambda, distance_M_i_1, 'gaussian')

            # compute target
            sv_target = torch.log(qi_1 + 10**(-10)) - torch.log(t_prev + 10**(-10))

            ''' Compute grad_theta '''
            # sub-sampling
            batch = [ii % N0 for ii in range(t * batchsize, (t + 1) * batchsize)]
            theta = particles.detach().clone().numpy()
            for m in range(M):
                w1, b1, w2, b2, loggamma = unpack_weights(theta[m, :], d, n_hidden)
                dw1, db1, dw2, db2, dloggamma = logp_gradient(X_train[batch, :], y_train[batch], w1, b1, w2,
                                                                          b2, loggamma, loglambda, N0)
                grad_theta[m, :] = pack_weights(dw1, db1, dw2, db2, dloggamma)

            if t_prev.requires_grad:
                sv_target.backward(torch.FloatTensor(torch.ones(M)))
                grad_sv_target = torch.from_numpy(grad_theta).clone().float() + particles.grad.clone()
                particles.grad.zero_()

            # compute delta_theta used to update all particles
            delta_theta = (1/M) * (torch.mm(torch.from_numpy(kxy).clone().float(), grad_sv_target.float()) + torch.from_numpy(dxkxy).clone().float())

            if t == 0:
                sum_squared_grad = torch.pow(delta_theta.detach().clone(), 2)
            else:
                sum_squared_grad = betta * sum_squared_grad + (1-betta) * torch.pow(delta_theta.detach().clone(), 2)

            epsilon_svgd = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad))
            with torch.no_grad():
                particles = particles + epsilon_svgd * delta_theta.detach().clone()
            particles.requires_grad = True

        # End of first SVGD
        ''' Second SVGD loop: distillation of newly obtained approximate likelihood via local particles update'''
        particles.requires_grad = False
        particles_2 = local_particles.detach().clone()
        particles_2.requires_grad = True

        for t in range(nb_svgd_2):
            kxy, dxkxy = svgd_kernel(particles_2.detach().clone().numpy(), h=-1)

            # compute t_prev = t^(i-1), qi=q^(i) and qi_1 = q^(i-1)
            distance_M_t = pairwise_distances(M, particles_2.transpose(0, 1), local_particles.transpose(0, 1))
            t_prev = kde(M, num_vars, my_lambda, distance_M_t, 'gaussian')
            log_t_prev = torch.log(t_prev + 10**(-10))

            distance_M_i = pairwise_distances(M, particles_2.transpose(0, 1), particles.transpose(0, 1))
            qi = kde(M, num_vars, my_lambda, distance_M_i, 'gaussian')
            log_qi = torch.log(qi + 10**(-10))

            distance_M_i_1 = pairwise_distances(M, particles_2.transpose(0, 1), global_particles.transpose(0, 1))
            qi_1 = kde(M, num_vars, my_lambda, distance_M_i_1, 'gaussian')
            log_qi_1 = torch.log(qi_1 + 10**(-10))

            sv_target = log_qi - log_qi_1 + log_t_prev
            if t_prev.requires_grad:
                sv_target.backward(torch.FloatTensor(torch.ones(M)))
                grad_sv_target = particles_2.grad.clone()
                particles_2.grad.zero_()

            # compute delta_theta used to update all particles
            delta_theta = (1/M) * (torch.mm(torch.from_numpy(kxy).float(), grad_sv_target.float()) + torch.from_numpy(dxkxy).float())

            if t == 0:
                sum_squared_grad_2 = torch.pow(delta_theta.detach().clone(), 2)
            else:
                sum_squared_grad_2 = betta * sum_squared_grad_2 + (1-betta) * torch.pow(delta_theta.detach().clone(), 2)

            epsilon_svgd = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad_2))
            with torch.no_grad():
                particles_2 = particles_2 + epsilon_svgd * delta_theta.detach().clone()
            particles_2.requires_grad = True

    # evaluate newly found particles (could be also done at the server)
    curr_err = evaluation(M, particles.detach().clone().numpy(), d, n_hidden, nn_predict, X_test, y_test, mean_X_train, std_X_train, mean_y_train, std_y_train)
    return particles.detach().clone(), particles_2.detach().clone(), curr_err


def server(nb_devices, particles, d, n_hidden, M, nb_svgd, nb_svgd_2, nb_global, y, X, y_test, X_test, batchsize):
    """
    Function that simulates the central server and schedules agents in a round robin fashion
    """

    # initialize local particles buffer at each device
    local_particles = particles.repeat(nb_devices, 1, 1)

    err = np.zeros(nb_global)  # array of RMSE across comm. rounds

    for i in range(0, nb_global):

        # schedule one device per global iteration in a round robin manner
        curr_id = i % nb_devices + 1
        kk = curr_id - 1
        # local datasets of scheduled device
        X_curr, y_curr = X[kk * X.shape[0] // nb_devices: ((kk + 1) * X.shape[0]) // nb_devices, :],\
                         y[kk * X.shape[0] // nb_devices: ((kk + 1) * X.shape[0]) // nb_devices]

        particles, local_particles[kk], err[i] = \
            agent_dsvgd(i, d, n_hidden, M, nb_svgd, nb_svgd_2, y_curr, X_curr, y_test, X_test, particles, local_particles[kk], batchsize=batchsize)

    return err


if __name__ == '__main__':
    ''' Parameters'''
    torch.random.manual_seed(0)
    np.random.seed(0)
    alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
    nb_svgd = 200  # number of iterations for first SVGD loop with target \tilde{p}
    nb_svgd_2 = 200  # number of iterations for second SVGD loop with target t
    nb_global = 21  # number of global iterations
    my_lambda = 0.55  # bandwidth for kde
    a0 = 1  # gamma distribution first parameter
    b0 = 0.1  # rate=1/scale: second parameter of gamma distribution
    alpha_ada = 10 ** (-3)  # constant rate used in AdaGrad
    epsilon_ada = 10 ** (-6)  # fudge factor for adagrad
    betta = 0.9  # for momentum update
    nb_exp = 2  # number of random trials to average upon
    M = 20  # number of particles
    n_hidden = 50  # Number of hidden neurons: 100 for year and 50 for other datasets
    nb_devices = 2  # Number of agents
    train_ratio = 0.9
    batchsize = 100
    array_acc = np.zeros(nb_global)
    loglambda = 1

    ''' load data file '''
    data = np.loadtxt('../data/kin8nm.txt')
    # TODO: Please make sure that Last column is the label and the other columns are the features
    X_input = data[:, range(data.shape[1] - 1)]
    y_input = data[:, data.shape[1] - 1]

    d = X_input.shape[1]  # dimension of each particle = dimensions of a data point

    for exp in range(nb_exp):
        print('Trial ', exp + 1)
        permutation = np.arange(X_input.shape[0])
        random.shuffle(permutation)

        size_train = int(np.round(X_input.shape[0] * train_ratio))
        index_train = permutation[0: size_train]
        index_test = permutation[size_train:]

        ''' build the training and testing data set'''
        X_train, y_train = X_input[index_train, :], y_input[index_train]
        X_test, y_test = X_input[index_test, :], y_input[index_test]

        d = X_train.shape[1]  # dimension of each particle = dimensions of a data point
        num_vars = d * n_hidden + n_hidden * 2 + 2  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 1 variance

        ''' Normalize dataset '''
        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0] = 1
        mean_X_train = np.mean(X_train, 0)

        mean_y_train = np.mean(y_train)
        std_y_train = np.std(y_train)
        X_train, y_train = normalization(X_train, y_train, mean_X_train, std_X_train, mean_y_train, std_y_train)

        ''' Initialize particles'''
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

        particles = torch.from_numpy(theta)

        ''' Run DSVGD server with Round Robin (RR) scheduling '''
        curr_acc = server(nb_devices, particles, d, n_hidden, M, nb_svgd, nb_svgd_2, nb_global, y_train, X_train, y_test, X_test, batchsize)
        array_acc += curr_acc

    print('BNN regression accuracy with DSVGD as function of comm. rounds =  ', repr(array_acc / nb_exp))







