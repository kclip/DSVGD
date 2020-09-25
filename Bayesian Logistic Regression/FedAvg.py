import torch
from torch.distributions.normal import Normal
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.distributions.multivariate_normal import MultivariateNormal
from Library.general_functions import sigmoid
from Library.bayesian_logistic_regression import BayesianLR


def fed_avg_client(alpha_ada, betta, epsilon_ada, theta, nb_iter, X, y, batchsize):

    # sum of gradients used in AdaGrad
    sum_squared_grad = 0
    for t in range(nb_iter):

        batch = [i % X.shape[0] for i in range(t * batchsize, (t + 1) * batchsize)]
        ridx = np.random.permutation(batch)

        Xs = X[ridx, :]
        ys = y[ridx]

        # compute gradient
        A = ys * np.matmul(theta, Xs.T)
        A = (- (1 - sigmoid(A)) * ys)

        delta_theta = X.shape[0] / Xs.shape[0] * (A.T * Xs).sum(axis=0)  # re-scaled gradient

        if t == 0:
            sum_squared_grad = delta_theta ** 2
        else:
            sum_squared_grad = betta * sum_squared_grad + (1 - betta) * (delta_theta ** 2)

        ada_step = alpha_ada / (epsilon_ada + np.sqrt(sum_squared_grad))

        theta = theta - ada_step * delta_theta
    return theta


def server(alpha_ada, betta, epsilon_ada, a, b, theta, nb_iter, nb_global, K, y, X, y_test, X_test, batchsize):
    """
    Implements Federated Averaging (McMahan et al., 2017) with round robin scheduling
    of one worker/agent per global iteration
    """

    tot_size = X.shape[0]  # total size of datasets
    acc = np.zeros(nb_global)
    llh = np.zeros(nb_global)
    model = BayesianLR(X, y)
    for i in range(nb_global):

        # pick one agent in a round robin manner
        curr_agent = i % K

        # local training dataset
        X_curr, y_curr = X[curr_agent*X.shape[0]//K: ((curr_agent + 1)*X.shape[0]) // K, :], y[curr_agent*X.shape[0]//K : ((curr_agent + 1)*X.shape[0]) // K]
        n_curr = X_curr.shape[0]

        # update global parameters after re-weighting with respect to local dataset sizes of each agent
        theta = (n_curr/tot_size) * fed_avg_client(alpha_ada, betta, epsilon_ada, theta, nb_iter, X_curr, y_curr, batchsize) + (1-n_curr/tot_size) * theta

        acc[i], llh[i] = model.evaluation(theta, X_test, y_test)

    return acc, llh


if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)
    alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
    nb_exp = 10  # number of experiments to average upon
    avg_accuracy = 0  # average accuracy after iterating nb_exp times through dataset
    avg_llh = 0  # average log-likelihood after iterating nb_exp times through dataset
    nb_global = 100  # number of global iterations
    nb_iter = 200  # number of local iterations per client
    K = 20  # number of agents
    alpha_ada = 0.05  # constant rate used in AdaGrad
    epsilon_ada = 10 ** (-9)  # fudge factor for adagrad
    N = 1  # number of particles = 1 as it is a frequentist approach
    batchsize = 100  # size of a batch
    betta = 0.9  # for momentum update
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

        ''' Run Federated Averaging server with Round Robin (RR) scheduling '''
        curr_accuracy, curr_llh =\
            server(alpha_ada, betta, epsilon_ada, a, b, particles.detach().numpy(), nb_iter, nb_global, K, y_train, X_train, y_test, X_test, batchsize)

        array_accuracy += curr_accuracy
        array_llh += curr_llh

    print('BLR accuracy with FedAvg as function of comm. rounds = ', repr(array_accuracy / nb_exp))
    print('BLR average llh with FedAvg as function of comm. rounds = ', repr(array_llh / nb_exp))