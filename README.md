# Federated Generalized Bayesian Learning via Distributed Stein Variational Gradient Descent (DSVGD)

Distributed Stein Variational Gradient Descent (DSVGD), is a non-parametric generalized Bayesian inference framework for federated learning. DSVGD maintains a number of non-random and interacting particles at a central server to represent the current iterate of the model global posterior. The particles are iteratively downloaded and updated by one of the agents with the end goal of minimizing the global free energy. By varying the number of particles, DSVGD enables a flexible trade-off between per-iteration communication load and number of communication rounds. DSVGD is shown to compare favorably to benchmark frequentist and Bayesian federated learning strategies while also providing well-calibrated, and hence trustworthy, predictions.


This repository contains the code for the experiments of the paper:
[Federated Generalized Bayesian Learning via Distributed Stein Variational Gradient Descent](https://arxiv.org/pdf/2009.06419.pdf).

## Toy Example: 1-D Mixture of Gaussians
 *Will be added soon.*

## Required Packages
 *Will be added soon.*

## Usage
 *Will be added soon.*

## Citation
If you find this repository helpful, please consider citing [our paper](https://arxiv.org/pdf/2009.06419.pdf) using the following BibTeX entry:
```
@article{kassab2020federated,
  title={Federated Generalized Bayesian Learning via Distributed Stein Variational Gradient Descent},
  author={Kassab, Rahif and Simeone, Osvaldo},
  journal={arXiv preprint arXiv:2009.06419},
  year={2020}
}
```

## Feedback
If you have any questions, suggestions or would like to discuss or collaborate, feel free to [drop me an email](rahif.kassab@kcl.ac.uk).
