# Sampling normalizing flows.

Simple code to sample a posterior for some model parameters $\pi=(A, B)$ of a model for the power spectrum $P(k) = Ak^{-B}$ of a one-dimensional Gaussian random field (GRF).

Fit the flow, sample the posterior with a uniform prior on $\pi$ given a measurement $\hat{\xi}$ of a GRF using an MCMC sampler.

![alt text](mcmc.png?raw=true)