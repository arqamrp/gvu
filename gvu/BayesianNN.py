import torch
import torch.nn as nn
import torch.nn.functional as F

from BayesianLinear import BayesianLinear

class BayesianNN(nn.Module):
    def __init__(self, model_init_config):
        super().__init__()        

        layers_config = model_init_config['layers']
        self.layers = nn.ModuleList()
        for i in range(len(layers_config) - 1):
            self.layers.append(BayesianLinear(layers_config[i], layers_config[i + 1], model_init_config))        

        self.posterior_log_prob = None
        self.fisher_info = None

        
    def forward(self, x, sample = True, prior = False, frozen_prior = False, reuse= False):
        x = x.view(-1, self.layers[0].in_features)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu( layer(x, sample = sample, prior = prior, frozen_prior = frozen_prior, reuse = reuse))        
        x = self.layers[-1](x, sample = sample, prior = prior, frozen_prior = frozen_prior, reuse = reuse)
        return x
    
    def posterior_log_prob(self):
        return sum(layer.posterior_log_prob() for layer in self.layers)
            # self.prior_log_probs = sum(layer.prior_log_probs for layer in self.layers)
            # self.frozen_prior_log_probs = sum(layer.frozen_prior_log_probs for layer in self.layers)

         # unnnormalised logits

    def freeze(self):
        for layer in self.layers:
            layer.freeze()
        print('Parameters frozen for unlearning prior successfully!')
    
    def divergence(self, unlearn = False, div_type = None, alpha = None):
        return sum(layer.divergence(unlearn = unlearn, div_type = div_type, alpha = alpha) for layer in self.layers)
    
    def max_log_probs_frozen_prior(self):
        return sum(layer.max_log_probs_frozen_prior() for layer in self.layers)
    