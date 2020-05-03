import torch
from torch.optim.optimizer import Optimizer

class pSGLD(Optimizer):
    """Implements pSGLD algorithm based on https://arxiv.org/pdf/1512.07666.pdf

    Built on the PyTorch RMSprop implementation 
    (https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html#RMSprop)
    """

    def __init__(self, params, lr=1e-2, beta=0.99, Lambda=1e-15, weight_decay=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= Lambda:
            raise ValueError("Invalid epsilon value: {}".format(Lambda))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr, beta=beta, Lambda=Lambda, centered=centered, 
            weight_decay=weight_decay)
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('pSGLD does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                V = state['V']
                beta = group['beta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                V.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(beta).add_(1 - beta, grad)
                    G = V.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['Lambda'])
                else:
                    G = V.sqrt().add_(group['Lambda'])

                p.data.addcdiv_(grad, G, value=-group['lr'])

                noise_std = 2*group['lr']/G
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0, std=1)*noise_std
                p.data.add_(noise)

        return G