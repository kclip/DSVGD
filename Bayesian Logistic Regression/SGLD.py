import torch
from torch.distributions.normal import Normal
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from torch.distributions.multivariate_normal import MultivariateNormal
from Library.bayesian_logistic_regression import BayesianLR
"""
    Bayesian Logistic Regression using SGLD (Welling & Teh, 2011).
    We use similar setting to Gershman et al., 2012.
"""


def sgld(a_0, a, b, theta, N, nb_iter, y_train, X_train, y_test, X_test, batchsize):
    """
    Implements Stochastic Gradient Langevin Dynamics (Welling & Teh, 2011) for Bayesian Logistic Regression
    """
    # initialize Bayesian Logistic Regression Model
    model = BayesianLR(X_train, y_train, batchsize, a, b)

    for t in range(nb_iter):
        epsilon_t = a_0 * (0.5 + t) ** (-0.55)
        theta = theta + (1/N) * (epsilon_t * 0.5 * model.dlnprob(theta) + np.random.normal(0, epsilon_t ** 2))

    return model.evaluation(theta, X_test, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
    nb_iter = 2000  # number of iterations
    epsilon_ada = 10 ** (-9)  # fudge factor for adagrad
    betta = 0.9  # for momentum update
    nb_exp = 50  # number of experiments to average upon
    N = 6  # number of particles
    avg_accuracy = 0  # average accuracy after iterating nb_exp times through dataset
    avg_llh = 0  # average log-likelihood after iterating nb_exp times through dataset
    a_0 = 0.01  # initial learning rate for SGLD, fixed via validation
    batchsize = 100

    # read data points from covertype dataset
    data = scipy.io.loadmat('../data/covertype.mat')
    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = -1  # please ensure labels are in {-1, +1}

    d = X_input.shape[1]  # dimension of each particle = dimensions of a data point

    avg_accuracy = 0
    avg_llh = 0
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

        ''' Run SGLD '''
        curr_accuracy, curr_llh = sgld(a_0, a, b, particles.detach().numpy(), N, nb_iter, y_train, X_train, y_test, X_test, batchsize)

        avg_accuracy += curr_accuracy
        avg_llh += curr_llh

    print('BLR average accuracy with SGLD = ', repr(avg_accuracy / nb_exp))
    print('BLR average llh with SGLD = ', repr(avg_llh / nb_exp))

