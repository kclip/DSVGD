import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

"""
    This file contains general purpose functions used across different experiments
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pairwise_distances(N, x, y):
    """
    Returns distance matrix that contains the distance between each particle (column) in x and y
    """
    x_squared = x.norm(dim=0)**2
    new_x = x_squared.view(1, -1).repeat(N, 1).transpose(0, 1)
    y_squared = y.norm(dim=0)**2
    new_y = y_squared.view(1, -1).repeat(N, 1)
    return new_x + new_y - 2*torch.mm(x.transpose(0, 1), y)


def rbf_vectorized(distances_squared):
    """
    Returns rbf evaluated at each possible pair of particles
    """
    # compute variable bandwidth h = med^2/n of the RBF kernel
    n = len(distances_squared)
    h = (np.median(distances_squared.detach().numpy()))/np.log(n)
    return torch.exp(-distances_squared / h)


def kde(N, d, my_lambda, distances_squared, kernel='gaussian'):
    """
    KDE over d dimensional particles
    """
    if kernel == 'gaussian':
        exp_distances = torch.exp(distances_squared*(-0.5)*(1/my_lambda**2)) + 10**(-9)
        sum_exp_distances = exp_distances.sum(dim=1)
        # return sum_exp_distances/(N*(np.pi*2*my_lambda**2)**(d/2))
        # use the log to avoid overflow in the denominator when d is large
        return torch.exp(torch.log(sum_exp_distances + 10**(-50)) - (d/2)*np.log(N*(np.pi*2*my_lambda**2) + 10**(-50)))


def svgd_kernel(theta, h=-1):
    """
    This function is borrowed from SVGD original paper code accessible at
    https://github.com/DartML/Stein-Variational-Gradient-Descent.
    Returns RBF kernel matrix and its derivative
    """
    sq_dist = pdist(theta)
    pairwise_dists = squareform(sq_dist) ** 2
    if h < 0:  # if h < 0, using median trick
        h = np.median(pairwise_dists)
        h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

    # compute the rbf kernel
    Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

    dxkxy = -np.matmul(Kxy, theta)
    sumkxy = np.sum(Kxy, axis=1)
    for i in range(theta.shape[1]):
        dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
    dxkxy = dxkxy / (h ** 2)
    return (Kxy, dxkxy)