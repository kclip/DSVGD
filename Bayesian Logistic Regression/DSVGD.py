import torch
from torch.distributions.normal import Normal
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.distributions.multivariate_normal import MultivariateNormal
from Library.bayesian_logistic_regression import BayesianLR
from Library.general_functions import pairwise_distances, kde, svgd_kernel


def agent_dsvgd(i, d, N, nb_svgd, nb_svgd_2, dlnprob, global_particles, local_particles):
    """
    Device running two SVGD loops to update local and global particles
    """
    sum_squared_grad = torch.zeros([N, d + 1])  # sum of gradient squared to use in Ada grad learning rate for first SVGD loop
    sum_squared_grad_2 = torch.zeros([N, d + 1])  # sum of gradient squared to use in Ada grad learning rate for second SVGD loop

    if i == 0:
        # It's the first iteration for each agent, set q^(i-1) = q^(0) when i = 0
        particles = global_particles.detach().clone()
        particles.requires_grad = False
        particles_2 = local_particles.detach().clone()
        particles_2.requires_grad = False
    else:
        # download particles from previous PVI iteration
        particles = global_particles.detach().clone()
        particles.requires_grad = True

        ''' First SVGD loop: update of global particles'''
        for t in range(0, nb_svgd):

            # calculating the kernel matrix and its derivative
            kxy, dxkxy = svgd_kernel(particles.detach().clone().numpy(), h=-1)

            # compute t_prev = t^(i-1)
            distance_M_j = pairwise_distances(N, particles.transpose(0, 1), local_particles.transpose(0, 1))
            t_prev = kde(N, d, my_lambda, distance_M_j, 'gaussian')

            # compute qi_1 = q^(i-1)
            distance_M_i_1 = pairwise_distances(N, particles.transpose(0, 1), global_particles.transpose(0, 1))
            qi_1 = kde(N, d, my_lambda, distance_M_i_1, 'gaussian')

            # compute SVGD target and its gradient
            sv_target = torch.log(qi_1 + 10 ** (-10)) - torch.log(t_prev + 10 ** (-10))
            sv_target.backward(torch.FloatTensor(torch.ones(N)))
            grad_sv_target = torch.from_numpy(dlnprob(particles.clone().detach().numpy())).float() + \
                             particles.grad.clone()
            particles.grad.zero_()

            # compute delta_theta used to update all particles
            delta_theta = (1 / N) * (torch.mm(torch.from_numpy(kxy).float(), grad_sv_target) + torch.from_numpy(dxkxy).float())

            # add delta_theta to gradient history
            if t == 0:
                sum_squared_grad = torch.pow(delta_theta.detach().clone(), 2)
            else:
                sum_squared_grad = betta * sum_squared_grad + (1 - betta) * torch.pow(
                    delta_theta.detach().clone(), 2)

            # Adagrad with momentum learning rate
            ada_step = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad))
            # update global particles
            with torch.no_grad():
                particles = particles + ada_step * delta_theta.detach().clone()
            particles.requires_grad = True
        # End of first SVGD loop

        ''' Second SVGD loop: distillation of newly obtained approximate likelihood via local particles update'''
        particles.requires_grad = False
        particles_2 = local_particles.detach().clone()
        particles_2.requires_grad = True

        for t in range(nb_svgd_2):

            # calculating the kernel matrix and its derivative
            kxy, dxkxy = svgd_kernel(particles_2.detach().clone().numpy(), h=-1)

            # compute t_prev = t^(i-1), qi=q^(i) and qi_1 = q^(i-1)
            distance_M_t = pairwise_distances(N, particles_2.transpose(0, 1), local_particles.transpose(0, 1))
            t_prev = kde(N, d, my_lambda, distance_M_t, 'gaussian')
            log_t_prev = torch.log(t_prev + 10 ** (-10))

            distance_M_i = pairwise_distances(N, particles_2.transpose(0, 1), particles.transpose(0, 1))
            qi = kde(N, d, my_lambda, distance_M_i, 'gaussian')
            log_qi = torch.log(qi + 10 ** (-10))

            distance_M_i_1 = pairwise_distances(N, particles_2.transpose(0, 1), global_particles.transpose(0, 1))
            qi_1 = kde(N, d, my_lambda, distance_M_i_1, 'gaussian')
            log_qi_1 = torch.log(qi_1 + 10 ** (-10))

            # compute SVGD target and its gradient
            sv_target = log_qi - log_qi_1 + log_t_prev
            sv_target.backward(torch.FloatTensor(torch.ones(N)))
            grad_sv_target = particles_2.grad.clone()
            particles_2.grad.zero_()

            # compute delta_theta used to update all particles
            delta_theta = (1 / N) * (torch.mm(torch.from_numpy(kxy).float(), grad_sv_target) + torch.from_numpy(dxkxy).float())

            if t == 0:
                sum_squared_grad_2 = torch.pow(delta_theta.detach().clone(), 2)
            else:
                sum_squared_grad_2 = betta * sum_squared_grad_2 + (1 - betta) * torch.pow(
                    delta_theta.detach().clone(), 2)

            ada_step = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad_2))
            # update local particles
            with torch.no_grad():
                particles_2 = particles_2 + ada_step * delta_theta.detach().clone()
            particles_2.requires_grad = True
        # END of distillation

    # return updated global particles to be used in the next pvi iteration and downloaded by other devices and local particles
    # to be used when calling the current agent again
    return particles.detach().clone(), particles_2.detach().clone()


def server_RR(nb_devices, particles, d, N, nb_svgd, nb_svgd_2, nb_global, etta, y, X, y_test, X_test, batchsize):
    """
    Server that schedules one agent out of nb_devices agents
    in a round robin fashion
    """
    # arrays containing acc and llh across communication rounds
    array_acc = np.zeros(nb_global)
    array_llh = np.zeros(nb_global)

    # initialize globlal particles
    particles = torch.cat((particles.detach().clone(), torch.log(etta.detach().clone())), 1)  # (N x d+1) tensor

    # initialize local particles buffers at each device
    local_particles = particles.repeat(nb_devices, 1, 1)

    # model that contains gradient of logistic function for each device over its own local dataset
    my_models = [0] * nb_devices
    for k in range(nb_devices):
        my_models[k] = BayesianLR(X[k*X.shape[0]//nb_devices: ((k + 1)*X.shape[0]) // nb_devices, :],\
                                  y[k*X.shape[0]//nb_devices: ((k + 1)*X.shape[0]) // nb_devices], batchsize, a, b)

    for i in range(0, nb_global):
        # scheduled device
        curr_id = i % nb_devices + 1
        kk = curr_id - 1

        # get new global particles updated by the agent and save local particles in a buffer for agent kk
        particles, local_particles[kk] = agent_dsvgd(i, d, N, nb_svgd, nb_svgd_2,\
                                     my_models[kk].dlnprob, particles, local_particles[kk])

        # evaluate accuracy and llh
        array_acc[i], array_llh[i] = my_models[kk].evaluation(particles.clone().detach().numpy(), X_test, y_test)

    return array_acc, array_llh


if __name__ == '__main__':
    # Parameters
    torch.random.manual_seed(0)
    np.random.seed(0)
    alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
    nb_svgd = 200  # number of iterations for svgd with target \tilde{p}
    nb_svgd_2 = 200  # number of iterations for svgd with target approximate likelihood t
    my_lambda = 0.55  # bandwidth for kde_vectorized
    alpha_ada = 0.05  # constant rate used in AdaGrad
    epsilon_ada = 10 ** (-9)  # fudge factor for adagrad
    betta = 0.9  # for momentum update
    nb_exp = 10  # number of random trials to average upon
    avg_accuracy = 0  # average accuracy after iterating nb_exp times through dataset
    avg_llh = 0  # average log-likelihood after iterating nb_exp times through dataset
    batchsize = 100
    nb_devices = 2  # number of agents
    N = 6  # number of particles
    nb_global = 11  # number of global iterations = I + 1 (first iteration is for initalization)
    average_accuracy_pvi = np.zeros(nb_global)
    average_llh_pvi = np.zeros(nb_global)

    ''' Read and preprocess data from covertype dataset'''
    data = scipy.io.loadmat('../data/covertype.mat')
    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = -1  # please ensure labels are in {-1, +1}

    d = X_input.shape[1]  # dimension of each particle = dimension of a data point

    for exp in range(nb_exp):
        print('Trial ', exp + 1)

        # split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)

        a, b = 1., 1 / 0.01  # b = rate = 1/scale
        p_etta = torch.distributions.gamma.Gamma(torch.tensor([a]), torch.tensor([b]))
        etta = p_etta.rsample(torch.Size([N]))
        etta.requires_grad = True

        # initialize particles using mutlivariate normal
        particles = torch.zeros(torch.Size([N, d]))
        mean_0 = torch.zeros(d)
        for i in range(N):
            particles[i, :] = MultivariateNormal(mean_0, 1 / etta[i] * torch.eye(d, d)).rsample()

        ''' Run DSVGD server with Round Robin (RR) scheduling '''
        curr_accuracy, curr_llh = \
            server_RR(nb_devices, particles, d, N, nb_svgd, nb_svgd_2, nb_global, etta, y_train, X_train, y_test, X_test, batchsize)

        average_accuracy_pvi += curr_accuracy
        average_llh_pvi += curr_llh

    print('BLR average accuracy with DSVGD as function of comm. rounds = ', repr(average_accuracy_pvi[1::] / nb_exp))
    print('BLR average llh with DSVGD as function of comm. rounds = ', repr(average_llh_pvi[1::] / nb_exp))







