import torch
import torch.nn.functional as F


def sample_model_losses(model, x, target, samples, device, prior_loss_weight = 0, unlearn = False, classification = True, div_type = None, alpha = None):
        
        target_dims = model.layers[-1].out_features
        batch_size = x.shape[0]
        
        # holding tensors
        preds = torch.zeros(samples, batch_size, target_dims).to(device)
        prior_preds = torch.zeros(samples, batch_size, target_dims).to(device)
        
        # filling them in
        for i in range(samples):
            if prior_loss_weight != 0 and unlearn:
                prior_preds[i] = model.forward(x, prior = True)
            preds[i] = model.forward(x, sample = True)
        
        # compute negative log likelihood
        if classification:
            pred_probs = F.softmax(preds, dim = 1)      # probabilities from unnormalised logits
            pred_probs_mean = pred_probs.mean(0)        # mean of probabilities across samples
            target_probs = F.one_hot(target, num_classes= target_dims)
            negative_log_likelihood = (-torch.sum(target_probs * torch.log(pred_probs_mean), dim=-1)).mean()
        else:
            pred_mean = preds.mean(0)
            negative_log_likelihood = (torch.sum( (pred_mean - target)**2, dim = -1)).mean()

        
        # compute prior predictive loss
        if prior_loss_weight != 0 and unlearn:
            if classification:
                prior_pred_probs = F.softmax(prior_preds, dim = 1) # probabilities from unnormalised logits
                prior_pred_probs_mean = prior_pred_probs.mean(0) # mean across samples
                prior_pred_mean_loss = (-torch.sum(prior_pred_probs_mean * torch.log(pred_probs_mean), dim=-1)).mean()

                centered_pred_probs = pred_probs - pred_probs_mean.unsqueeze(0)
                centered_prior_pred_probs = prior_pred_probs - prior_pred_probs_mean.unsqueeze(0)

                pred_probs_cov = torch.einsum('sbn,sbm->bnm', centered_pred_probs, centered_pred_probs) / (pred_probs.size(0) - 1)
                prior_pred_probs_cov = torch.einsum('sbn,sbm->bnm', centered_prior_pred_probs, centered_prior_pred_probs) / (prior_pred_probs.size(0) - 1)
                prior_pred_cov_loss = torch.linalg.matrix_norm( prior_pred_probs_cov - pred_probs_cov, ord ='fro', dim = (1, 2)).mean()
            else:
                prior_pred_mean = prior_preds.mean(0) # mean across samples
                prior_pred_mean_loss = (torch.sum( (pred_mean - prior_pred_mean )**2, dim = -1)).mean()

                centered_preds = preds - pred_mean.unsqueeze(0)
                centered_prior_preds = prior_preds - prior_pred_mean.unsqueeze(0)

                pred_cov = torch.einsum('sbn,sbm->bnm', centered_preds, centered_preds) / (preds.size(0) - 1)
                prior_pred_cov = torch.einsum('sbn,sbm->bnm', centered_prior_preds, centered_prior_preds) / (prior_preds.size(0) - 1)
                prior_pred_cov_loss = torch.linalg.matrix_norm( prior_pred_cov - pred_cov, ord ='fro', dim = (1, 2)).mean()
        else:
            prior_pred_mean_loss = None
            prior_pred_cov_loss = None
        
        divergence = model.divergence(unlearn = unlearn, div_type = div_type, alpha = alpha)
        
        return {
            'prior_regularisation_term' : divergence,
            'negative_log_likelihood': negative_log_likelihood,
            'prior_pred_mean_loss' : prior_pred_mean_loss,
            'prior_pred_cov_loss': prior_pred_cov_loss
        }