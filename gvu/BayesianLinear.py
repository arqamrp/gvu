import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

from distributions import Gaussian

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, model_init_config):
        super().__init__()
        
        seed = model_init_config['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) 
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(model_init_config['init_mu_low'], model_init_config['init_mu_high']))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(model_init_config['init_rho_low'], model_init_config['init_rho_high']))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        self.weight_prior_mu = nn.Parameter(model_init_config['prior_mu'] * torch.ones(out_features, in_features) , requires_grad = False)
        self.weight_prior_rho = nn.Parameter(torch.log(torch.exp(torch.tensor(model_init_config['prior_sigma'])) - 1) * torch.ones(out_features, in_features) , requires_grad = False)
        self.weight_prior = Gaussian(self.weight_prior_mu, self.weight_prior_rho)
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(model_init_config['init_mu_low'], model_init_config['init_mu_high']))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(model_init_config['init_rho_low'], model_init_config['init_rho_high']))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)        

        self.bias_prior_mu = nn.Parameter(model_init_config['prior_mu'] * torch.ones(out_features, ) , requires_grad = False)
        self.bias_prior_rho = nn.Parameter(torch.log(torch.exp(torch.tensor(model_init_config['prior_sigma'])) - 1) * torch.ones(out_features, ) , requires_grad = False)
        self.bias_prior = Gaussian(self.bias_prior_mu, self.bias_prior_rho)
        
        self.posterior_log_prob = None

        self.weight_frozen_prior_mu = None
        self.weight_frozen_prior_rho = None
        self.bias_frozen_prior_mu = None
        self.bias_frozen_prior_rho = None

        self.last_sampled_weight = None
        self.last_sampled_bias = None
    
    def forward(self, x, sample = False, prior = False, frozen_prior = False, reuse = False):
        device = self.weight_prior_mu.device
        if reuse:
            w = self.last_sampled_weight
            b = self.last_sampled_bias
        elif prior:
            w = self.weight_prior.sample(device)
            b = self.bias_prior.sample(device)
        elif frozen_prior:
            w = self.weight_frozen_prior.sample(device)
            b = self.bias_frozen_prior.sample(device)
        elif sample:
            w = self.weight.sample(device)
            b = self.bias.sample(device)
        else:
            w = self.weight.mu
            b = self.bias.mu
        
        self.last_sampled_weight = w
        self.last_sampled_bias = b
        return F.linear(x, w, b)

        def posterior_log_prob(self):
            return self.weight.log_prob(self.last_sampled_weight) + self.bias.log_prob(self.last_sampled_bias)
        
        # self.prior_log_probs = self.weight_prior.log_prob(w) + self.bias_prior.log_prob(b)
        # self.frozen_prior_log_probs = self.weight_frozen_prior.log_prob(w) + self.bias_frozen_prior.log_prob(b)

    def freeze(self):
        # deepcopy current params to store as a new prior
        self.weight_frozen_prior_mu = nn.Parameter(deepcopy(self.weight_mu.detach()), requires_grad=False)
        self.weight_frozen_prior_rho = nn.Parameter(deepcopy(self.weight_rho.detach()), requires_grad=False)
        self.bias_frozen_prior_mu = nn.Parameter(deepcopy(self.bias_mu.detach()), requires_grad=False)
        self.bias_frozen_prior_rho = nn.Parameter(deepcopy(self.bias_rho.detach()), requires_grad=False)
        
        self.weight_frozen_prior = Gaussian(self.weight_frozen_prior_mu, self.weight_frozen_prior_rho)
        self.bias_frozen_prior = Gaussian(self.bias_frozen_prior_mu, self.bias_frozen_prior_rho)

    def max_log_prob_frozen_prior(self):
        return self.weight_frozen_prior.log_prob(self.weight_frozen_prior_mu) + self.bias_frozen_prior.log_prob(self.bias_frozen_prior_mu)
    
    def divergence(self, unlearn = False, div_type = None, alpha = None):        
        # for unlearning use frozen params from full training as priors
        if unlearn:
            weight_prior_mu = self.weight_frozen_prior.mu
            weight_prior_sigma = self.weight_frozen_prior.sigma
            bias_prior_mu = self.bias_frozen_prior.mu
            bias_prior_sigma = self.bias_frozen_prior.sigma
        # for training simply use N(0,1) priors
        elif not unlearn: 
            weight_prior_mu = self.weight_prior.mu
            weight_prior_sigma = self.weight_prior.sigma
            bias_prior_mu = self.bias_prior.mu
            bias_prior_sigma = self.bias_prior.sigma
            
        if div_type is None:
            div_type = 'RKL'
        if alpha is None: 
            alpha = 0
            
        # Forward KL divergence
        if div_type == "KL":
            weight_div = 0.5 * (weight_prior_mu - self.weight.mu)**2/(self.weight.sigma)**2 
            weight_div += 0.5 * (weight_prior_sigma **2 - self.weight.sigma**2)/(self.weight.sigma)**2
            weight_div += torch.log(self.weight.sigma/weight_prior_sigma)
            weight_div = weight_div.sum()
            
            bias_div = 0.5 * (bias_prior_mu - self.bias.mu)**2/(self.bias.sigma)**2 
            bias_div += (bias_prior_sigma **2 - self.bias.sigma**2)/(self.bias.sigma)**2 
            bias_div += torch.log(self.bias.sigma/bias_prior_sigma)
            bias_div = bias_div.sum()
            
            return weight_div + bias_div
        
        # Reverse KL Divergence
        elif div_type == "RKL":
            weight_div = 0.5 * ( weight_prior_mu - self.weight.mu )**2/(weight_prior_sigma)**2 
            weight_div += 0.5 * (self.weight.sigma**2 - weight_prior_sigma**2)/(weight_prior_sigma)**2
            weight_div += torch.log(weight_prior_sigma/self.weight.sigma)
            weight_div = weight_div.sum()
            
            bias_div = 0.5 * (bias_prior_mu - self.bias.mu )**2/(bias_prior_sigma)**2 
            bias_div += 0.5* (self.bias.sigma**2 - bias_prior_sigma**2)/(bias_prior_sigma)**2
            bias_div += torch.log(bias_prior_sigma/self.bias.sigma)
            bias_div = bias_div.sum()
            
            return weight_div + bias_div
            
        # Fisher Distance 
        elif div_type == "F":
            weight_fisher = (self.weight.mu - weight_prior_mu)**2 + 2 * (weight_prior_sigma - self.weight.sigma)**2
            weight_fisher *= ( (self.weight.mu - weight_prior_mu)**2 + 2 * (weight_prior_sigma + self.weight.sigma)**2 )
            weight_fisher = torch.sqrt(weight_fisher)            
            
            weight_div = weight_fisher + (self.weight.mu - weight_prior_mu)**2 + 2 * (weight_prior_sigma**2 + self.weight.sigma**2)
            weight_div /= 4 * weight_prior_sigma * self.weight.sigma
            weight_div = math.sqrt(2) * torch.log(weight_div).sum()
            
            bias_fisher = (self.bias.mu - bias_prior_mu)**2 + 2 * (bias_prior_sigma - self.bias.sigma)**2
            bias_fisher *= ( (self.bias.mu - bias_prior_mu)**2 + 2 * (bias_prior_sigma + self.bias.sigma)**2 )
            bias_fisher = torch.sqrt(bias_fisher) 
            
            bias_div = bias_fisher + (self.bias.mu - bias_prior_mu)**2 + 2 * (bias_prior_sigma**2 + self.bias.sigma**2)
            bias_div /= 4 * bias_prior_sigma * self.bias.sigma 
            bias_div = math.sqrt(2) * torch.log(bias_div).sum()
            
            return weight_div + bias_div

        
        # Jensen-Shannon Divergence
        elif div_type == "JS":
            weight_midpoint_sigma =  0.5 * torch.sqrt(self.weight.sigma**2 + weight_prior_sigma**2)
            weight_midpoint_mu = 0.5 * (self.weight.mu + weight_prior_mu)
            # KL(p||m)
            weight_div = 0.5 * (weight_prior_mu - weight_midpoint_mu)**2/(weight_midpoint_sigma)**2
            weight_div += 0.5 * (weight_prior_sigma**2 - weight_midpoint_sigma**2)/(weight_midpoint_sigma)**2 
            weight_div += torch.log(weight_midpoint_mu/weight_prior_sigma)
            # KL(q||m)
            weight_div += 0.5 * (self.weight.mu - weight_midpoint_mu)**2 / (weight_midpoint_sigma)**2
            weight_div += 0.5 * (self.weight.sigma**2 - weight_midpoint_mu**2)/(weight_midpoint_mu)**2
            weight_div += torch.log(weight_midpoint_mu/self.weight.sigma)
            weight_div = weight_div.sum()
            
            bias_midpoint_sigma =  0.5 * torch.sqrt(self.bias.sigma**2 + bias_prior_sigma**2)
            bias_midpoint_mu = 0.5 * (self.bias.mu + bias_prior_mu)
            # KL(p||m)
            bias_div = 0.5 * (bias_prior_mu - bias_midpoint_mu)**2/(bias_midpoint_sigma)**2 
            bias_div += 0.5 * (bias_prior_sigma**2 - bias_midpoint_sigma**2)/(bias_midpoint_sigma)**2
            bias_div += torch.log(bias_midpoint_mu/bias_prior_sigma)
            # KL(q||m)
            bias_div += 0.5 * (self.bias.mu - bias_midpoint_mu)**2 / (bias_midpoint_sigma)**2 
            bias_div += 0.5 * (self.bias.sigma**2 - bias_midpoint_mu**2)/(bias_midpoint_mu)**2
            bias_div += torch.log(bias_midpoint_mu/self.bias.sigma).sum()
            
            return 0.5 * (weight_div + bias_div)
        
        # Alpha-Renyi Divergence
        elif div_type == "AR":
            weight_sigma2_a = alpha * weight_prior_sigma**2 + (1-alpha) * self.weight.sigma**2
            weight_div = torch.log(weight_prior_sigma/ self.weight.sigma)
            weight_div += 0.5 * torch.log(weight_prior_sigma/weight_sigma2_a) / (alpha- 1)
            weight_div += 0.5 * alpha * (self.weight.mu - weight_prior_mu)**2 / weight_sigma2_a
            weight_div = weight_div.sum()

            bias_sigma2_a = alpha * bias_prior_sigma**2 + (1-alpha) * self.bias.sigma**2
            bias_div = torch.log(bias_prior_sigma / self.bias.sigma)
            bias_div += 0.5 * torch.log(bias_prior_sigma/bias_sigma2_a) / (alpha- 1) 
            bias_div += 0.5 * alpha * (self.bias.mu - bias_prior_mu)**2 / bias_sigma2_a
            bias_div = bias_div.sum()
         
            return weight_div + bias_div

        # Alpha Divergence
        elif div_type == "A":
            weight_sigma2_a = alpha* weight_prior_sigma**2 + (1-alpha)* self.weight.sigma**2
            weight_frac = (weight_prior_sigma**alpha) * (self.weight.sigma**(1-alpha)) / torch.sqrt(weight_sigma2_a)
            weight_exp  = torch.exp( - 0.5 * (alpha) * (1-alpha) * (self.weight.mu - weight_prior_mu)**2 / weight_sigma2_a )
            weight_div =  (1- weight_frac * weight_exp).sum() / (alpha*(1-alpha))
            
            bias_sigma2_a = alpha* bias_prior_sigma**2 + (1-alpha)* self.bias.sigma**2
            bias_frac = (bias_prior_sigma**alpha) * (self.bias.sigma**(1-alpha)) / torch.sqrt(bias_sigma2_a)
            bias_exp  = torch.exp( - 0.5 * (alpha) * (1-alpha) * (self.bias.mu - bias_prior_mu)**2 / bias_sigma2_a )
            bias_div =  (1- bias_frac * bias_exp).sum() / (alpha*(1-alpha))
            
            return weight_div + bias_div

        # Hellinger Distance 
        elif div_type == "HEL":
            # this one is redirected at layer declaration to div_type="A", alpha=0.5
            return self.divergence(unlearn = unlearn, div_type = 'A', alpha = 0.5)

#         # Pearson Divergence
        elif div_type == "PEAR":
            return self.divergence(unlearn = unlearn, div_type = 'A', alpha = 2)
        
        else:
            raise ValueError(f"Invalid divergence type: {div_type}")