import torch
from langevin_sampling.samplers import LangevinDynamics
import numpy as np
import matplotlib.pyplot as plt
from rosenbrock import rosenbrock
import copy
from tqdm import tqdm

np.random.seed(10)
torch.manual_seed(10)

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')


def rosenbrock_negative_log(x):
    return rosen_dist.nl_pdf(x.unsqueeze(0))


if __name__ == '__main__':

    # Initialize parameters
    mu = torch.tensor([0.0], device=device)
    a = torch.tensor([.2], device=device)
    b = torch.ones([2, 1], device=device)

    # Define the distribution
    rosen_dist = rosenbrock.RosenbrockDistribution(mu, a, b, device=device)

    x = torch.randn([2], requires_grad=True, device=device)
    max_itr = int(1e5)
    langevin_dynamics = LangevinDynamics(x,
                                         rosenbrock_negative_log,
                                         lr=2.5,
                                         lr_final=1e-2,
                                         max_itr=max_itr,
                                         device=device)

    hist_samples = []
    loss_log = []
    for j in tqdm(range(max_itr)):
        est, loss = langevin_dynamics.sample()
        loss_log.append(loss)
        if j % 10 == 0:
            hist_samples.append(est.cpu().numpy())
    est_samples = np.array(hist_samples)[500:]

    num_samples = est_samples.shape[0]
    true_samples = rosen_dist.sample(num_samples).cpu().numpy()

    fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
    plt.plot(loss_log)
    plt.title("Unnormalized PDF")
    plt.xlabel("Iterations")
    plt.ylabel(r"$- \log \ p_X(\mathbf{x}) + const.$")
    plt.grid()

    fig = plt.figure(dpi=150, figsize=(9, 4))
    plt.subplot(121)
    plt.scatter(est_samples[:, 0], est_samples[:, 1], s=.5, color="#db76bf")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([-6, 6])
    plt.ylim([-3, 25])
    plt.title("Langevin dynamics")
    plt.subplot(122)
    p2 = plt.scatter(true_samples[:, 0],
                     true_samples[:, 1],
                     s=.5,
                     color="#5e838f")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([-6, 6])
    plt.ylim([-3, 25])
    plt.title(r"$\mathbf{x} \sim p_X(\mathbf{x})$")
    plt.tight_layout()
    plt.show()
