import torch
from langevin_dynamics import LangevinDynamics
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

class BananaDistribution(object):
    def __init__(self, c, device='cuda'):

        super(BananaDistribution, self).__init__()
        self.c = c
        self.normal = torch.distributions.normal.Normal(torch.zeros_like(c), 
            torch.ones_like(c))

    def nl_pdf(self, x):
        z1 = self.c[0]*x[0]
        z2 = x[1]/self.c[0]-self.c[1]*self.c[0]**2*(x[0]**2+1.)
        return 0.5*(z1**2+z2**2)

    def sample(self):
        z = self.normal.sample()
        x = torch.zeros([2], device=self.c.device)
        x[0] = z[0]/self.c[0]
        x[1] = z[1]*self.c[0]+self.c[0]*self.c[1]*(z[0]**2+self.c[0]**2)
        return x


if __name__ == '__main__':

    banana_dist = BananaDistribution(c=torch.Tensor([1.0, 2.0]), device=device)

    x = torch.randn([2], requires_grad=True, device=device)
    max_itr = int(1e4)
    langevin_dynamics = LangevinDynamics(x, banana_dist.nl_pdf, lr=1e-1, lr_final=8e-2, 
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
        true_samples[j, :] = banana_dist.sample().cpu().numpy()

    fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
    plt.plot(loss_log); plt.title("Unnormalized PDF")
    plt.xlabel("Iterations")
    plt.ylabel(r"$- \log \ p_X(\mathbf{x}) + const.$")
    plt.grid()

    fig = plt.figure(dpi=150, figsize=(9, 4))
    plt.subplot(121); 
    plt.scatter(est_samples[:, 0], est_samples[:, 1], s=.5, color="#db76bf")
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    plt.xlim([-4, 4]); plt.ylim([-2, 25])
    plt.title("Langevin dynamics")
    plt.subplot(122); 
    p2 = plt.scatter(true_samples[:, 0], true_samples[:, 1], s=.5, color="#5e838f")
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
    plt.xlim([-4, 4]); plt.ylim([-2, 25])
    plt.title(r"$\mathbf{x} \sim p_X(\mathbf{x})$")
    plt.tight_layout()
    # plt.savefig('figs/banana-LD.png', format="png", bbox_inches="tight", dpi=300)
    plt.show()