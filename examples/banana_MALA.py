import torch
import numpy as np
import matplotlib.pyplot as plt
from langevin_sampling.samplers import *
from rosenbrock import *
from tqdm import tqdm
np.random.seed(19)
torch.manual_seed(19)

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def rosenbrock_negative_log(x):
    return rosen_dist.nl_pdf(x.unsqueeze(0))

if __name__ == '__main__':

    # Initialize parameters
    mu = torch.Tensor([0.0], device=device)
    a = torch.Tensor([.2], device=device)
    b = torch.ones([2, 1], device=device)

    # Define the distribution
    rosen_dist = RosenbrockDistribution(mu, a, b, device=device)

    x = torch.randn([2], requires_grad=True, device=device)
    max_itr = int(1e4)
    mala = MetropolisAdjustedLangevin(x, rosenbrock_negative_log, 
        lr=15e-1, lr_final=9e-1, max_itr=max_itr, device=device)

    hist_samples = []
    loss_log = []
    for j in tqdm(range(max_itr)):
        est, loss = mala.sample()
        loss_log.append(loss)
        if j%3 == 0:
            hist_samples.append(est.cpu().numpy())
    est_samples = np.array(hist_samples)[200:]

    num_samples = est_samples.shape[0]
    true_samples = rosen_dist.sample(num_samples).cpu().numpy()

    fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
    plt.plot(loss_log); plt.title("Unnormalized PDF")
    plt.xlabel("Iterations")
    plt.ylabel(r"$- \log \ p_X(\mathbf{x}) + const.$")
    plt.grid()

    fig = plt.figure(dpi=150, figsize=(9, 4))
    plt.subplot(121); 
    plt.scatter(est_samples[:, 0], est_samples[:, 1], s=.5, color="#db76bf")
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    plt.xlim([-6, 6]); plt.ylim([-3, 25])
    plt.title("Metropolis-adjusted Langevin dynamics")
    plt.subplot(122); 
    p2 = plt.scatter(true_samples[:, 0], true_samples[:, 1], s=.5, color="#5e838f")
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    plt.xlim([-6, 6]); plt.ylim([-3, 25])
    plt.title(r"$\mathbf{x} \sim p_X(\mathbf{x})$")
    plt.tight_layout()
    plt.savefig('fig/sample-rosenblock.png', format="png", bbox_inches="tight", dpi=300)
    plt.show()
