import torch
from torchvision import datasets, transforms
from generate_sets import generate_sets
from learn_unlearn import learn_unlearn
import os

def is_dataset_downloaded(dataset_path):
    return os.path.exists(dataset_path)

dataset_configs = [
    {
        'name': 'FMNIST',
        'classification': True,
        'target_dims':  10,

        'original_train_set': datasets.FashionMNIST(
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
            {0 : 5000, 1: 5000}, 
            {0 : 2500, 1: 2500, 2: 2500, 3: 2500}, 
            {i : 2000 for i in range(5)} ,
            {i : 1000 for i in range(10)} 
        ]
    },
    
    {
        'name': 'MNIST',
        'classification': True,
        'target_dims':  10,
        
        'original_train_set': datasets.MNIST(
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
            {0 : 5000, 1: 5000},
            {0 : 2500, 1: 2500, 2: 2500, 3: 2500}, 
            {i : 2000 for i in range(5)},
            {i : 1000 for i in range(10)} 
        ]
    }
]



model_init_config = {
    'layers': [784, 100, 100, 100, 10],
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
    'lr': 1e-4,
    
    'train_epochs': 200,
    'train_div_type': 'RKL',
    'train_alpha': None,
    
    'unlearn_epochs': 100,
    'prior_loss_weight': 0,
    'prior_cov_weight': 0,
    'unlearn_div_type': 'RKL',
    'unlearn_alpha': None    
}

test_config = {
    'batch_size': 1000,
    'sample_size': 10,
}

for dataset_config in dataset_configs:
    generate_sets(dataset_config)
    for forget_idx in range(len(dataset_config['forget_set_configs'])):
        learn_unlearn(dataset_config, forget_idx, learn_unlearn_config = learn_unlearn_config, model_init_config = model_init_config, retain= True, full_train = True, unlearn = True)