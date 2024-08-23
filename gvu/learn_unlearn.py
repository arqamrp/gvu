import os
import torch
from torch.utils.data import DataLoader

from BayesianNN import BayesianNN
from train_BNN import train_BNN
from unlearn_BNN import unlearn_BNN
import wandb



def learn_unlearn(dataset_config, forget_idx, learn_unlearn_config, model_init_config):
    # initialise
    device = learn_unlearn_config['device']
    name = dataset_config['name']
    div_type = learn_unlearn_config['div_type']
    opt = learn_unlearn_config['optimizer']

    plw = learn_unlearn_config['prior_loss_weight']
    adj_lam = learn_unlearn_config['adj_lam']
    # method = learn_unlearn_config['unlearn_method']

    
    project_id = f"GVU-{name}-{forget_idx}-{div_type}-{adj_lam}-{plw}"
    
    if not os.path.exists(f'ckp/{project_id}'):
        os.makedirs(f'ckp/{project_id}')
    
    full_train_set, val_set, forget_set, retain_set = dataset_config['full_train_set'], dataset_config['val_set'], dataset_config['forget_sets'][forget_idx][0], dataset_config['forget_sets'][forget_idx][1]
    val_loader = DataLoader(val_set, batch_size = learn_unlearn_config['batch_size'], shuffle = True)
    full_train_loader =  DataLoader(full_train_set, batch_size = learn_unlearn_config['batch_size'], shuffle = True)
    retain_loader =  DataLoader(retain_set, batch_size = learn_unlearn_config['batch_size'], shuffle = True)
    forget_loader =  DataLoader(forget_set, batch_size = learn_unlearn_config['batch_size'], shuffle = True)
    
    # train on retain set only
    if learn_unlearn_config['retain']:
        retain_path = f'ckp/{project_id}/retain'
        if not os.path.exists(retain_path):
            os.makedirs(retain_path)

        retain_model = BayesianNN(model_init_config).to(device)
        wandb.init(project= project_id+'-retain', config= learn_unlearn_config)
        wandb.watch(retain_model, log="all", log_freq=10)
        optimizer = opt(retain_model.parameters(), lr = learn_unlearn_config['lr'])
        train_BNN(retain_model, optimizer = optimizer, train_loader= retain_loader, val_loader = val_loader, train_config = learn_unlearn_config, path = retain_path)
        wandb.finish()
        retain_model.eval()

        torch.save(retain_model.state_dict(), f'{retain_path}.pth')
        print(f'Retain set model trained and saved to ckp/{project_id}/retain.pth')
    else: 
        retain_model = BayesianNN(model_init_config).to(device)
        retain_model.load_state_dict(torch.load(f'ckp/{project_id}/retain.pth'))
        
    # train on full train set
    if learn_unlearn_config['full_train']:
        full_path = f'ckp/{project_id}/full'
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        full_trainset_model = BayesianNN(model_init_config).to(device)
        wandb.init(project = project_id + '-full', config= learn_unlearn_config)
        wandb.watch(full_trainset_model, log="all", log_freq=10)
        optimizer = opt(full_trainset_model.parameters(), lr = learn_unlearn_config['lr'])
        train_BNN(full_trainset_model, optimizer = optimizer, train_loader= full_train_loader, val_loader = val_loader, train_config = learn_unlearn_config, path = full_path)
        wandb.finish()
        full_trainset_model.eval()
        torch.save(full_trainset_model.state_dict(), f'ckp/{project_id}/full.pth')
        print(f'Full train set model trained and saved to ckp/{project_id}/full.pth')
        full_trainset_model.freeze()
    else:
        full_trainset_model = BayesianNN(model_init_config).to(device)
        full_trainset_model.load_state_dict(torch.load(f'ckp/{project_id}/full.pth'))
        full_trainset_model.freeze()
    
    
    # unlearning on forget set
    if learn_unlearn_config['unlearn']:
        unlearn_path = f'ckp/{project_id}/unlearn'
        if not os.path.exists(unlearn_path):
            os.makedirs(unlearn_path)
        wandb.init(project = project_id + '-unlearn', config= learn_unlearn_config)
        wandb.watch(full_trainset_model, log="all", log_freq=10)
        optimizer = opt(full_trainset_model.parameters(), lr = learn_unlearn_config['lr'])
        unlearn_BNN(full_trainset_model, optimizer = optimizer, forget_loader= forget_loader, val_loader = val_loader, unlearn_config = learn_unlearn_config, retain_model = retain_model, path = unlearn_path)
        wandb.finish()
        full_trainset_model.eval()
        torch.save(full_trainset_model.state_dict(), f'ckp/{project_id}/unlearn.pth')
        print(f'Unlearning model trained and saved to ckp/{project_id}/unlearn.pth')
