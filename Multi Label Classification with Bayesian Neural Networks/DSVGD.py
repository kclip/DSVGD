import torch
import numpy as np
from SVGD_multidimensional import pairwise_distances, kde
import random
import theano.tensor as T
import theano
import pickle
from Library.bayesian_neural_networks_classification import pack_weights, unpack_weights, evaluation, init_weights
from Library.general_functions import svgd_kernel
"""
    Bayesian Neural network (BNN) for multi-label classification using DSVGD.
    We use similar setting to Hernandez-Lobato & Adams (2015) with the difference that we use a 
    categorical distribution to model the log-likelihood p(y|W, X) parameterized by the output of the BNN.
    We borrow some of the code from SVGD original paper available at: 
    https://github.com/DartML/Stein-Variational-Gradient-Descent  
    We fix the covariance of the prior on the weight to loglambda = 1.
"""


def agent_dsvgd(id, i, d, n_hidden, M, nb_svgd, nb_svgd_2, y_train, X_train, global_particles, local_particles, batchsize):
    sum_squared_grad = torch.zeros([M, num_vars])  # sum of gradient squared to use in Ada grad learning rate for first SVGD
    sum_squared_grad_2 = torch.zeros([M, num_vars])  # sum of gradient squared to use in Ada grad learning rate for second svgd

    ''' Define the neural network here '''
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

            # compute target distribution
            sv_target = torch.log(qi_1 + 10**(-10)) - torch.log(t_prev + 10**(-10))

            ''' Compute grad_theta '''
            # sub-sampling
            batch = [ii % N0 for ii in range(t * batchsize, (t + 1) * batchsize)]
            theta = particles.detach().clone().numpy()
            for m in range(M):
                w1, b1, w2, b2 = unpack_weights(theta[m, :], d, n_hidden, number_labels)
                dw1, db1, dw2, db2 = logp_gradient(X_train[batch, :], y_train[batch], w1, b1, w2, b2, loglambda, N0)
                grad_theta[m, :] = pack_weights(dw1, db1, dw2, db2)

            if t_prev.requires_grad:
                sv_target.backward(torch.ones(M, dtype=torch.double))
                grad_sv_target = torch.from_numpy(grad_theta).clone().double() + particles.grad.clone()
                particles.grad.zero_()

            # compute delta_theta used to update all particles
            if M != 1:
                delta_theta = (1/M) * (torch.mm(torch.from_numpy(kxy).clone().double(), grad_sv_target.double()) + torch.from_numpy(dxkxy).clone().double())
            else:
                delta_theta = grad_sv_target

            if t == 0:
                sum_squared_grad = torch.pow(delta_theta.detach().clone(), 2)
            else:
                sum_squared_grad = betta * sum_squared_grad + (1-betta) * torch.pow(delta_theta.detach().clone(), 2)

            epsilon_svgd = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad))
            # update global particles
            with torch.no_grad():
                particles = particles + epsilon_svgd * delta_theta.detach().clone()
            particles.requires_grad = True

        # End of first SVGD loop

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

            # compute target distribution
            sv_target = log_qi - log_qi_1 + log_t_prev
            if t_prev.requires_grad:
                sv_target.backward(torch.ones(M, dtype=torch.double))
                grad_sv_target = particles_2.grad.clone()
                particles_2.grad.zero_()
            else:
                print('Yooooo')
                grad_sv_target = 0
            # compute delta_theta used to update all particles
            delta_theta = (1/M) * (torch.mm(torch.from_numpy(kxy).double(), grad_sv_target.double()) + torch.from_numpy(dxkxy).double())

            if t == 0:
                sum_squared_grad_2 = torch.pow(delta_theta.detach().clone(), 2)
            else:
                sum_squared_grad_2 = betta * sum_squared_grad_2 + (1-betta) * torch.pow(delta_theta.detach().clone(), 2)

            epsilon_svgd = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad_2))
            # update local particles
            with torch.no_grad():
                particles_2 = particles_2 + epsilon_svgd * delta_theta.detach().clone()
            particles_2.requires_grad = True

    return particles.detach().clone(), particles_2.detach().clone(), nn_predict


def server(nb_devices, particles, d, n_hidden, M, nb_svgd, nb_svgd_2, nb_global, y, X, y_test, X_test, batchsize):
    """
    Function that simulates the central server and schedules agents in a roundrobin fashion
    """

    # initialize local particles at each device
    local_particles = particles.repeat(nb_devices, 1, 1)
    acc = np.zeros(nb_global)
    llh = np.zeros(nb_global)

    for i in range(0, nb_global):
        # schedule an agent in a RR fashion
        curr_id = i % nb_devices

        X_curr, y_curr = X[curr_id * X.shape[0] // nb_devices: ((curr_id + 1) * X.shape[0]) // nb_devices, :], y[curr_id * X.shape[
            0] // nb_devices: ((curr_id + 1) * X.shape[0]) // nb_devices]
        particles, local_particles[curr_id], nn_predict = agent_dsvgd(curr_id+1, i, d, n_hidden,\
        M, nb_svgd, nb_svgd_2, y_curr, X_curr, particles.double(), local_particles[curr_id], batchsize=batchsize)

        acc[i], llh[i] = evaluation(M, particles.detach().clone().numpy(), d, n_hidden, number_labels, nn_predict, X_test, y_test)
    return acc, llh


if __name__ == '__main__':
    ''' Parameters'''
    torch.random.manual_seed(0)
    np.random.seed(0)
    alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
    nb_svgd = 200  # number of iterations for first SVGD loop with target \tilde{p}
    nb_svgd_2 = 200  # number of iterations for second SVGD loop with target t
    nb_global = 11  # number of PVI iterations
    my_lambda = 0.55  # bandwidth for kde_vectorized
    number_labels = 10  # number of labels in MNIST
    alpha_ada = 10 ** (-3)  # constant rate used in AdaGrad
    epsilon_ada = 10 ** (-6)  # fudge factor for adagrad
    betta = 0.9  # for momentum update
    nb_exp = 1  # number of random trials
    nb_devices = 2  # number of agents
    train_ratio = 0.8
    batchsize = 100
    loglambda = 1  # log precision of weight prior
    n_hidden = 100  # number of neurons of the hidden layer
    M = 20  # number of particles
    array_acc = np.zeros(nb_global)
    array_llh = np.zeros(nb_global)

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
        for i in range(M):
            w1, b1, w2, b2 = init_weights(d, n_hidden, number_labels)
            theta[i, :] = pack_weights(w1, b1, w2, b2)

        particles = torch.from_numpy(theta).double()

        ''' Run DSVGD server with Round Robin (RR) scheduling '''
        curr_acc, cur_llh = server(nb_devices, particles, d, n_hidden, M, nb_svgd, nb_svgd_2, nb_global, y_train, X_train, y_test, X_test, batchsize)
        array_acc += curr_acc
        array_llh += cur_llh

    print('BNN multilabel classification accuracy with DSVGD as function of comm. rounds = ', repr(array_acc / nb_exp))
    print('BNN multilabel classification llh with DSVGD as function of comm. rounds = ', repr(array_llh / nb_exp))







