import torch
from torch.distributions.normal import Normal
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.distributions.multivariate_normal import MultivariateNormal
from Library.bayesian_logistic_regression import BayesianLR


def d_sgld_RR(a_0, a, b, theta, d, N, nb_iter, nb_global, K, y, X, y_test, X_test, batchsize):
    """
    Implements Distributed Stochastic Gradient Langevin Dynamics for Bayesian Logistic Regression by Ahn et al.
    Assumptions made:
        - Assumes response delay unchanged and same at all workers
        - Assumes trajectory length to be the same across all workers
    """
    arr_acc = np.zeros(nb_global)
    arr_llh = np.zeros(nb_global)

    # initialize Bayesian Logistic Regression Models, one per worker (equivalent to data shard)
    models = [0] * K
    for k in range(K):
        models[k] = BayesianLR(X[k*X.shape[0]//K : ((k + 1)*X.shape[0]) // K, :], y[k*X.shape[0]//K : ((k + 1)*X.shape[0]) // K], batchsize, a, b)

    for i in range(nb_global):
        # assign workers to chains
        if N > K:
            assignments = np.random.permutation(K)  # assignments[i] = worker sampling chain i
            chain = i % K

            # each chain corresponds to N//K particles
            theta[chain*(N//K) : N//K + chain*(N//K)] = sample_traj(i, a_0, theta[chain*(N//K) : N//K + chain*(N//K)], nb_iter, models[assignments[chain]])
        else:
            # in this case, one chain per randomly selected worker
            assignments = np.random.permutation(K)  # assignments[i] = worker sampling chain i
            chain = i % N

            # each chain corresponds to one particle
            theta[chain: chain + 1] = sample_traj(i, a_0, theta[chain: chain + 1], nb_iter, models[assignments[chain]])
        arr_acc[i], arr_llh[i] = models[assignments[chain]].evaluation(theta, X_test, y_test)

    return arr_acc, arr_llh


def sample_traj(i, a_0, theta, nb_iter, model):
    """
    samples a trajectory of length nb_iter corresponding to a chain starting at theta and on worker/model
    """
    for t in range(nb_iter):
        epsilon_t = a_0 * (0.5 + (i*nb_iter + t)) ** (-0.55)
        theta = theta + (1/len(theta)) * (epsilon_t * 0.5 * model.dlnprob(theta) + np.random.normal(0, epsilon_t ** 2))
    return theta


if __name__ == '__main__':
    np.random.seed(0)
    alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
    avg_accuracy = 0  # average accuracy after iterating nb_exp times through dataset
    avg_llh = 0  # average log-likelihood after iterating nb_exp times through dataset
    a_0 = 0.01  # initial learning rate for DSGLD, fixed via validation
    nb_exp = 10  # number of experiments to average upon
    batchsize = 100  # size of a batch
    nb_global = 100  # number of global iterations
    nb_iter = 200  # trajectory length per worker
    K = 20  # number of agents/workers
    N = 6  # number of praticles
    array_accuracy = np.zeros(nb_global)
    array_llh = np.zeros(nb_global)

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

        a, b = 1., 1/0.01  # b = rate = 1/scale
        p_etta = torch.distributions.gamma.Gamma(torch.tensor([a]), torch.tensor([b]))
        etta = p_etta.rsample(torch.Size([N]))

        # initialize particles using mutlivariate normal
        particles = torch.zeros(torch.Size([N, d]))
        mean_0 = torch.zeros(d)
        for i in range(N):
            particles[i, :] = MultivariateNormal(mean_0, 1/etta[i]*torch.eye(d, d)).rsample()

        particles = torch.cat((particles.detach().clone(), torch.log(etta.detach().clone())), 1)

        ''' Run DSGLD server with Round Robin (RR) scheduling '''
        curr_accuracy, curr_llh = d_sgld_RR(a_0, a, b, particles.detach().numpy(), d, N, nb_iter, nb_global, K, y_train, X_train, y_test, X_test, batchsize)

        array_accuracy += curr_accuracy
        array_llh += curr_llh

    print('BLR average accuracy with DSGLD as function of comm. rounds = ', repr(array_accuracy / nb_exp))
    print('BLR average llh with DSGLD as function of comm. rounds = ', repr(array_llh / nb_exp))