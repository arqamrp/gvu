import math
import torch


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.std_normal = torch.distributions.Normal(0, 1)


    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self, device):
        eps = self.std_normal.sample(self.rho.size()).to(device)
        return self.mu + self.sigma * eps

    def log_prob(self, x):
        return (- 0.5 * math.log(2*math.pi) - torch.log(self.sigma) - 0.5 * (x- self.mu)**2/self.sigma**2).sum()
