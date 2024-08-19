import torch
import numpy as np
from torch.utils.data import Subset, random_split

def generate_sets(dataset_config:dict, seed = 42):    
    dataset_name = dataset_config['name']
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
    print(f'Seed set as {seed}')

    
    # create validation set
    print(f'Generating validation set for {dataset_name}')
    original_train_set = dataset_config['original_train_set']
    dataset_config['full_train_size'] = len(original_train_set) - dataset_config['val_size']

    full_train_set, val_set = random_split(original_train_set, [dataset_config['full_train_size'], dataset_config['val_size']])
    full_train_set.classes = original_train_set.classes
    val_set.classes = original_train_set.classes
    
    print(f"Original non-test size: {len(original_train_set)}")
    print(f"Full training set size: {len(full_train_set)}")
    print(f"Validation set size: {len(val_set)}")

    dataset_config['val_set'] = val_set
    dataset_config['full_train_set'] = full_train_set
    class_indices = {i: [] for i in range(dataset_config['target_dims'])}
    
    dataset_config['forget_sets'] = {}
    
    # collate indices of training data points of different classes
    for idx, (img, label) in enumerate(full_train_set):
        class_indices[label].append(idx)
    
    for forget_set_idx, forget_config in enumerate(dataset_config['forget_set_configs']):
        print(f'Generating forget set {forget_set_idx} for {dataset_name}')

        # for each class sample the desired number of points for the forget set
        forget_indices = []
        for i in forget_config.keys():
            selected_indices = np.random.choice(class_indices[i], forget_config[i], replace=False)
            forget_indices.extend(selected_indices)

        forget_set = Subset(full_train_set, forget_indices)

        # create retain set by excluding forget set indices
        retain_indices = list(set(range(len(full_train_set))) - set(forget_indices))
        retain_set = Subset(full_train_set, retain_indices)

        print(f'Forget set size: {len(forget_set)}, retain set size: {len(retain_set)}') 
        dataset_config['forget_sets'][forget_set_idx] = (forget_set, retain_set)
        
    print(f'Finished forget sets for {dataset_name}\n')