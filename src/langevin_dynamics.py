import torch
from precondSGLD import pSGLD
import copy

class LangevinDynamics(object):
    def __init__(self, x, func, lr=1e-2, lr_final=1e-4, max_itr=1e4, device='cpu'):
        super(LangevinDynamics, self).__init__()
    
        self.x = x
        self.optim = pSGLD([self.x], lr, weight_decay=0.0)
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0

    def sample(self):
        self.lr_decay()
        loss = self.func(self.x)
        loss.backward()
        self.optim.step()
        self.counter += 1
        return copy.deepcopy(self.x.data), loss.item()

    def decay_fn(self, lr=1e-2, lr_final=1e-4, max_itr=1e4):
        gamma = -0.55
        b = max_itr/((lr_final/lr)**(1/gamma) - 1.0)
        a = lr/(b**gamma)
        def lr_fn(t, a=a, b=b, gamma=gamma):
            return a*((b + t)**gamma)
        return lr_fn

    def lr_decay(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_fn(self.counter)