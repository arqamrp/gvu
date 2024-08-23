import torch
import numpy as np
from torch.utils.data import Subset, random_split

def generate_classification_sets(dataset_config:dict, seed=42):    
    dataset_name = dataset_config['name']
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
    print(f'Seed set as {seed}')

    # create validation set from the test set
    print(f'Generating validation set for {dataset_name}')
    original_test_set = dataset_config['test_set']
    val_size = dataset_config['val_size']
    num_classes = dataset_config['target_dims']
    val_size_per_class = int(val_size/num_classes)
    
    class_indices = {i: [] for i in range(num_classes)}

    # Collate indices of test data points of different classes
    for idx, (img, label) in enumerate(original_test_set):
        class_indices[label].append(idx)
    
    # Select equal number of samples from each class for validation set
    val_indices = []
    for class_label, indices in class_indices.items():
        selected_indices = np.random.choice(indices, val_size_per_class, replace=False)
        val_indices.extend(selected_indices)

    val_set = Subset(original_test_set, val_indices)
    val_set.classes = original_test_set.classes
    
    test_indices = list(set(range(len(original_test_set))) - set(val_indices))
    test_set = Subset(original_test_set, test_indices)

    print(f"Validation set size: {len(val_set)}, {val_size_per_class} samples per class.")

    dataset_config['val_set'] = val_set
    dataset_config['test_set'] = test_set

    # Generate the training set without validation samples
    full_train_set = dataset_config['train_set']
    dataset_config['full_train_size'] = len(full_train_set)
    dataset_config['forget_sets'] = {}
    
    # collate indices of training data points of different classes
    train_class_indices = {i: [] for i in range(num_classes)}
    for idx, (img, label) in enumerate(full_train_set):
        train_class_indices[label].append(idx)
    
    for forget_set_idx, forget_config in enumerate(dataset_config['forget_set_configs']):
        print(f'Generating forget set {forget_set_idx} for {dataset_name}')

        # for each class sample the desired number of points for the forget set
        forget_indices = []
        for i in forget_config.keys():
            selected_indices = np.random.choice(train_class_indices[i], forget_config[i], replace=False)
            forget_indices.extend(selected_indices)

        forget_set = Subset(full_train_set, forget_indices)

        # create retain set by excluding forget set indices
        retain_indices = list(set(range(len(full_train_set))) - set(forget_indices))
        retain_set = Subset(full_train_set, retain_indices)

        print(f'Forget set size: {len(forget_set)}, retain set size: {len(retain_set)}') 
        dataset_config['forget_sets'][forget_set_idx] = (forget_set, retain_set)
        
    print(f'Finished forget sets for {dataset_name}\n')


