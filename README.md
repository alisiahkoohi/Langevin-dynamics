# Sampling with gradient-based Markov Chain Monte Carlo approaches

PyTorch implementation of [stochastic gradient Langevin dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) (SGLD) and  [preconditioned SGLD](https://arxiv.org/pdf/1512.07666.pdf) (pSGLD), involving simple examples of using unadjusted Langevin dynamics and [Metropolis-adjusted Langevin algorithm](https://link.springer.com/article/10.1023/A:1023562417138) (MALA) to sample from a 2D Gaussian distribution and "banana" distribution.

![](examples/fig/sample-rosenblock.png)

## Prerequisites

Follow the steps below to install the necessary libraries:

```bash
pip install langevin-sampling
```

## Script descriptions

`langevin_sampling/SGLD.py`: SGLD sampler.

`langevin_sampling/precondSGLD.py`: pSGLD sampler.

`langevin_sampling/samplers.py`:

* Implements `LangevinDynamics` class that given negative-log of unnormalized density function and starting guess, runs Langevin dynamics to sample from the given density.

* Implements `MetropolisAdjustedLangevin` class that given negative-log of unnormalized density function and starting guess, runs MALA to sample from the given density.

`examples/gaussian_LD.py`: Sampling from a toy 2D Gaussian distribution with unadjusted Langevin dynamics.

`examples/gaussian_MALA.py`: Sampling from a toy 2D Gaussian distribution with MALA.

`examples/banana_LD.py`: Sampling from a toy "banana" distribution with unadjusted Langevin dynamics.

`examples/banana_MALA.py`: Sampling from a toy "banana" distribution with MALA.

### Running the code

To sample a toy 2D Gaussian distribution with unadjusted Langevin dynamics, run:

```bash
python examples/gaussian_LD.py
```

To sample a toy 2D Gaussian distribution with MALA, run:

```bash
python examples/gaussian_MALA.py
```

To sample a toy "banana" distribution with unadjusted Langevin dynamics, run:

```bash
python examples/banana_LD.py
```

To sample a toy "banana" distribution with MALA, run:

```bash
python examples/banana_MALA.py
```


## References

* Max Welling and Yee Whye Teh. Bayesian Learning via Stochastic Gradient Langevin Dynamics. In Proceedings of the 28th International Conference on Machine Learning, ICML’11, pages 681–688, Madison, WI, USA, 2011. Omnipress. ISBN 9781450306195. doi: 10.5555/3104482.3104568. URL [https://dl.acm.org/doi/abs/10.5555/3104482.3104568](https://dl.acm.org/doi/abs/10.5555/3104482.3104568).

* Chunyuan Li, Changyou Chen, David Carlson, and Lawrence Carin. Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, AAAI’16, pages 1788–1794. AAAI Press, 2016. doi: 10.5555/3016100.3016149. URL [https://dl.acm.org/doi/abs/10.5555/3016100.3016149](https://dl.acm.org/doi/abs/10.5555/3016100.3016149).

* Roberts, G.O., Stramer, O. Langevin Diffusions and Metropolis-Hastings Algorithms. Methodology and Computing in Applied Probability 4, pages 337–357, 2002. doi: 10.1023/A:1023562417138. URL  [https://link.springer.com/article/10.1023/A:1023562417138](https://link.springer.com/article/10.1023/A:1023562417138).

* Pagani F, Wiegand M, Nadarajah S. An n-dimensional Rosenbrock distribution for MCMC testing. arXiv preprint arXiv:1903.09556. 2019 Mar 22. URL [https://arxiv.org/abs/1903.09556](https://arxiv.org/abs/1903.09556).

## Questions

Please contact alisk@rice.edu for further questions.


## Author

Ali Siahkoohi
