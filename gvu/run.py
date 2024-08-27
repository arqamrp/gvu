import torch
from torchvision import datasets, transforms

from generate_sets import generate_classification_sets
from learn_unlearn import learn_unlearn

import os
import argparse


def is_dataset_downloaded(dataset_path):
    return os.path.exists(dataset_path)

# Set up argument parser
parser = argparse.ArgumentParser(description='Train and Unlearn Models with Custom Configuration')

parser.add_argument('--retain', type = bool, default = True, help = 'Whether to train on the retain set')
parser.add_argument('--full_train', type = bool, default = True, help = 'Whether to train on the full train set')
parser.add_argument('--unlearn', type = bool, default = True, help = 'Whether to unlearn')

# parser.add_argument('--unlearn_method', type=str, default=None, help='Method for unlearning. Options: [eubo, pm]')

parser.add_argument('--div_type', type=str, default=None, help='Divergence type for training (default: RKL)')
parser.add_argument('--alpha', type=float, default=None, help='Alpha value for training divergence (if applicable)')

parser.add_argument('--adj_lam', type=float, default=0., help='Likelihood adjustment threshold')
parser.add_argument('--plw', type=float, default=0., help='Prior loss weight')

parser.add_argument('--fset', default = None, type=int, help='Forget set index')  # Forget set index
parser.add_argument('--dataset', default = None, type=str, help='Dataset name')  # Dataset index

# parser.add_argument('--unlearn_div_type', type=str, default= None, help='Divergence type for unlearning (default: RKL)')
# parser.add_argument('--unlearn_alpha', type=float, default=None, help='Alpha value for unlearning divergence (if applicable)')


# Options:

if __name__ == '__main__':
        args = parser.parse_args()

        # if args.train_div_type is not None and args.unlearn_div_type is not None:
        #     assert args.train_div_type == args.unlearn_div_type, 'Train and unlearn divergence types must match'
        #     assert args.train_alpha == args.unlearn_alpha, 'Train and unlearn alpha values must match'
        # elif args.train_div_type is not None and args.unlearn_div_type is None:
        #     args.unlearn_div_type = args.train_div_type
        #     args.unlearn_alpha = args.train_alpha


        dataset_configs = {
            'FMNIST': {
                'name': 'FMNIST',
                'classification': True,
                'target_dims':  10,

                'full_train_set': datasets.FashionMNIST(
                    './fmnist', 
                    train=True, 
                    download=not is_dataset_downloaded('./fmnist/raw/train-images-idx3-ubyte.gz'), 
                    transform=transforms.ToTensor()
                ),
                'test_set': datasets.FashionMNIST(
                    './fmnist', 
                    train=False, 
                    download=not is_dataset_downloaded('./fmnist/raw/t10k-images-idx3-ubyte.gz'), 
                    transform=transforms.ToTensor()
                ),

                'val_size': 5000,
                'forget_set_configs': [
                    {i : 1000 for i in range(10)},
                    {i : 2000 for i in range(5)},
                    {0 : 2500, 1: 2500, 2: 2500, 3: 2500},
                    {0 : 5000, 1: 5000}
                ]
            },

            'MNIST': {
                'name': 'MNIST',
                'classification': True,
                'target_dims':  10,
                
                'full_train_set': datasets.MNIST(
                    './mnist', 
                    train=True, 
                    download=not is_dataset_downloaded('./mnist/raw/train-images-idx3-ubyte.gz'), 
                    transform=transforms.ToTensor()
                ),
                'test_set': datasets.MNIST(
                    './mnist', 
                    train=False, 
                    download=not is_dataset_downloaded('./mnist/raw/t10k-images-idx3-ubyte.gz'), 
                    transform=transforms.ToTensor()
                ),
                
                'val_size': 5000,
                'forget_set_configs': [
                    {i : 1000 for i in range(10)},
                    {i : 2000 for i in range(5)},
                    {0 : 2500, 1: 2500, 2: 2500, 3: 2500},
                    {0 : 5000, 1: 5000}
                ]
            }
        }

        model_init_config = {
            'layers': [784, 128, 10],
            'seed': 42,
            'init_mu_low': -0.2,
            'init_mu_high': 0.2,
            'init_rho_low': -3,
            'init_rho_high': -2,
            
            'prior_mu': 0,
            'prior_sigma': 1    
        }

        learn_unlearn_config = {
            'batch_size': 1000,
            'sample_size': 10,
            'loader_kwargs': {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {},
            'device': torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'),
            'optimizer': torch.optim.Adam,
            'lr': 1e-3,
            
            'train_epochs': 200,
            
            'div_type': args.div_type,
            'alpha': args.alpha,
            
            'unlearn_epochs': 100,
            'prior_loss_weight': args.plw,
            # 'prior_cov_weight': 0,


            'retain': args.retain,
            'full_train': args.full_train,
            'unlearn': args.unlearn,

            'adj_lam': args.adj_lam
        }

        test_config = {
            'batch_size': 1000,
            'sample_size': 10,
        }

        if args.dataset is None and args.fset is None:
            for dataset_config in dataset_configs:
                generate_classification_sets(dataset_config)
                for forget_idx in range(len(dataset_config['forget_set_configs'])):
                    learn_unlearn(dataset_config, forget_idx, learn_unlearn_config=learn_unlearn_config, model_init_config=model_init_config)

        elif args.dataset is not None:
            dataset_config = dataset_configs[args.dataset]
            generate_classification_sets(dataset_config)
            if args.fset is not None:
                learn_unlearn(dataset_config, args.fset, learn_unlearn_config=learn_unlearn_config, model_init_config=model_init_config)
            else:
                for forget_idx in range(len(dataset_config['forget_set_configs'])):
                    learn_unlearn(dataset_config, forget_idx, learn_unlearn_config=learn_unlearn_config, model_init_config=model_init_config)
                 
                 