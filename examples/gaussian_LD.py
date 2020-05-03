import torch
from langevin_sampling.samplers import *
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
np.random.seed(19)
torch.manual_seed(19)

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class GaussianDistribution(object):
    def __init__(self, mu, cov, device='cuda'):
        super(GaussianDistribution, self).__init__()
        
        self.mu = mu
        self.cov = cov
        self.precision = torch.inverse(cov)

        self.R = torch.cholesky(self.cov)
        self.normal = torch.distributions.normal.Normal(torch.zeros_like(mu), 
            torch.ones_like(mu))

    def nl_pdf(self, x):
        return 0.5*(((x - self.mu).T).matmul(self.precision)).matmul(x - self.mu)

    def sample(self):
        return self.R.matmul(self.normal.sample()) + self.mu 


if __name__ == '__main__':

    dim = 2

    mu = torch.Tensor([1.2, .6], device=device)
    cov = 0.9*(torch.ones([2, 2], device=device) - torch.eye(2, device=device)).T + \
        torch.eye(2, device=device)*1.3
    gaussian_dist = GaussianDistribution(mu, cov, device=device)

    x = torch.zeros([2], requires_grad=True, device=device)
    max_itr = int(1e4)
    langevin_dynamics = LangevinDynamics(x, gaussian_dist.nl_pdf, lr=1e-1, lr_final=4e-2, 
        max_itr=max_itr, device=device)

    hist_samples = []
    loss_log = []
    for j in tqdm(range(max_itr)):
        est, loss = langevin_dynamics.sample()
        loss_log.append(loss)
        if j%3 == 0:
            hist_samples.append(est.cpu().numpy())
    est_samples = np.array(hist_samples)[200:]

    num_samples = est_samples.shape[0]
    true_samples = np.zeros([num_samples, 2])
    for j in range(num_samples):
        true_samples[j, :] = gaussian_dist.sample().cpu().numpy()

    fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
    plt.plot(loss_log); plt.title("Unnormalized PDF")
    plt.xlabel("Iterations")
    plt.ylabel(r"$- \log \ \mathrm{N}(\mathbf{x} | \mu, \Sigma) + const.$")
    plt.grid()

    fig = plt.figure(dpi=150, figsize=(9, 4))
    plt.subplot(121); 
    plt.scatter(est_samples[:, 0], est_samples[:, 1], s=.5, color="#db76bf")
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    plt.xlim([-3, 6]); plt.ylim([-4, 5])
    plt.title("Langevin dynamics")
    plt.subplot(122); 
    p2 = plt.scatter(true_samples[:, 0], true_samples[:, 1], s=.5, color="#5e838f")
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    plt.xlim([-3, 6]); plt.ylim([-4, 5])
    plt.title(r"$\mathbf{x} \sim \mathrm{N}(\mu, \Sigma)$")
    plt.tight_layout()
    plt.show()