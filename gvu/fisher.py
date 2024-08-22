# import torch
# import torch.nn.functional as F

# def compute_fisher_info_dataloader(model, data_loader, num_samples=None, device='cuda'):
    
#     fisher_info = {}
#     for name, param in model.named_parameters():
#         fisher_info[name] = torch.zeros_like(param, device=device)

#     model.eval()

#     for batch_idx, (x, y) in enumerate(data_loader):
#         if num_samples is not None and batch_idx >= num_samples:
#             break
        
#         x, y = x.to(device), y.to(device)

#         output = model(x, sample=True)
#         pred_probs = F.softmax(output, dim=1)

#         # Compute Fisher Information for each class label
#         for label_index in range(output.shape[1]):
#             label = torch.full((x.size(0),), label_index, dtype=torch.long).to(device)
#             negloglikelihood = F.cross_entropy(output, label)
            
#             # Zero out gradients from the previous pass
#             model.zero_grad()

#             # Backpropagate the NLL to obtain gradients
#             negloglikelihood.backward(retain_graph=True if (label_index + 1) < output.shape[1] else False)

#             # Accumulate the Fisher information
#             for name, param in model.named_parameters():
#                 if param.grad is not None:
#                     fisher_info[name] += (pred_probs[:, label_index] * (param.grad.detach() ** 2)).sum(dim=0)

#     # Normalize the Fisher information by the number of data points processed
#     total_samples = len(data_loader.dataset) if num_samples is None else min(num_samples, len(data_loader.dataset))
#     fisher_info = {name: value / total_samples for name, value in fisher_info.items()}

#     # Return the Fisher information dictionary
#     return fisher_info
