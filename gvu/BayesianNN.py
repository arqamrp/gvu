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
        
    def forward(self, x, sample = True, prior = False):
        x = x.view(-1, self.layers[0].in_features)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu( layer(x, sample = sample, prior = prior))        
        x = self.layers[-1](x, sample = sample, prior = prior)
        return x # unnnormalised logits

    def freeze(self):
        for layer in self.layers:
            layer.freeze()
        print('Parameters frozen for unlearning prior successfully!')
    
    def divergence(self, unlearn = False, div_type = None, alpha = None):
        return sum(layer.divergence(unlearn = unlearn, div_type = div_type, alpha = alpha) for layer in self.layers)