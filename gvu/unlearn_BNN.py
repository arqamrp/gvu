import torch
import pandas
import wandb
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F

from sample_model_losses import sample_model_losses

def unlearn_BNN(model, optimizer, forget_loader, val_loader, unlearn_config, retain_model, path, verbose=False):
    method = unlearn_config['unlearn_method']

    num_batches = len(forget_loader)
    forget_set_size = len(forget_loader.dataset)
    val_set_size = len(val_loader.dataset)
    batch_size = unlearn_config['batch_size']
    samples = unlearn_config['sample_size']
    n_epochs = unlearn_config['unlearn_epochs']
    device = unlearn_config['device']
    
    model.train()
    if verbose:
        print(f'Unlearning model on {forget_set_size} samples')
    

    # Define custom metrics to ensure correct step tracking
    wandb.define_metric("val/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")
     
    wandb.define_metric("trace/depoch")
    wandb.define_metric("trace/*", step_metric="trace/depoch")
    
    wandb.define_metric("unlearn/global_step")
    wandb.define_metric("unlearn/*", step_metric="unlearn/global_step")

    for epoch in tqdm(range(0, n_epochs)):

        for batch_idx, (data, target) in enumerate(forget_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            
            results = sample_model_losses(
                model, data, target, samples=samples, device=device,
                prior_loss_weight=unlearn_config['prior_loss_weight'], unlearn=True,
                classification=True, div_type=unlearn_config['unlearn_div_type']
            )
            
            if results['prior_pred_cov_loss'] is None:
                results['prior_pred_cov_loss'] = 0
            if results['prior_pred_mean_loss'] is None:
                results['prior_pred_mean_loss'] = 0
            
            gvo = - (1 - unlearn_config['prior_loss_weight']) * results['negative_log_likelihood']
            gvo += unlearn_config['prior_loss_weight'] * unlearn_config['prior_cov_weight'] * results['prior_pred_cov_loss']
            gvo += unlearn_config['prior_loss_weight'] * (1 - unlearn_config['prior_cov_weight']) * results['prior_pred_mean_loss']
            gvo += results['prior_regularisation_term'] / forget_set_size

            results['generalized_variational_objective'] = gvo
            
            if verbose:
                print(results)
            
            gvo.backward()
            optimizer.step()
            
            # Log to wandb without using `global_step`
            wandb.log({'unlearn/global_step': epoch * num_batches + batch_idx, 
                       **{'unlearn/'+key: results[key].item() for key in results if results[key] !=0} })
            
        if torch.isnan(gvo).any():
            print(f'NaN detected in GVO at epoch {epoch}, batch {batch_idx}')
            print(f'GVO: {gvo}')
            break  # Or handle it as needed

        model.eval()

        val_nll = 0
        retain_nll = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                val_results = sample_model_losses(model, data, target, samples=samples, device=device)
                val_nll += val_results['negative_log_likelihood'].item()
                
                outputs = model(data, sample=False)
                retain_outputs = retain_model(data, sample=False)
                pred_prob = F.softmax(outputs, dim=-1)
                retain_prob = F.softmax(retain_outputs, dim=-1)
                retain_nll += (-torch.sum(retain_prob * torch.log(pred_prob), dim=-1)).mean()
                
        val_nll = val_nll / len(val_loader)
        
        retain_nll = retain_nll / len(val_loader)
        
        # Log validation NLL and retain NLL to wandb
        wandb.log({'val/epoch': epoch,  'val/nll': val_nll, 'val/retain_nll': retain_nll})
            
        # Log histograms of the model weights and biases
        for i, layer in enumerate(model.layers):
            if not (torch.isnan(layer.weight_mu).any() or torch.isnan(layer.weight_rho).any() or 
                    torch.isnan(layer.bias_mu).any() or torch.isnan(layer.bias_rho).any()):
                wandb.log({
                    f'val/w{i}_mu': wandb.Histogram(layer.weight_mu.cpu().detach().numpy()),
                    f'val/w{i}_rho': wandb.Histogram(layer.weight_rho.cpu().detach().numpy()),
                    f'val/b{i}_mu': wandb.Histogram(layer.bias_mu.cpu().detach().numpy()),
                    f'val/b{i}_rho': wandb.Histogram(layer.bias_rho.cpu().detach().numpy())
                })
            else:
                print(f'Skipping logging for layer {i} due to NaN values.')

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f'{path}/{epoch}.pth')
            
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print({key: results[key].item() for key in results if results[key] !=0 })
            
            print(f'Epoch {epoch} validation NLL: {val_nll}')
            
            # Evaluation on the test set
            model.eval()
            all_labels = []
            unlearn_preds = []
            retain_preds = []
            retain_nll = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images, sample=False)
                    retain_outputs = retain_model(images, sample=False)
                    
                    _, preds = torch.max(outputs, 1)
                    _, retain_pred = torch.max(retain_outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    unlearn_preds.extend(preds.cpu().numpy())
                    retain_preds.extend(retain_pred.cpu().numpy())

            # Compute overall accuracy and concurrence with retain model
            accuracy = accuracy_score(all_labels, unlearn_preds)
            concurrence = accuracy_score(retain_preds, unlearn_preds)
            
            print(f'Overall Accuracy: {accuracy:.4f}')
            print(f'Concurrence with retain model: {concurrence:.4f}')

            # Compute precision, recall, and F1 score for each class
            class_names = val_loader.dataset.classes
            report = classification_report(all_labels, unlearn_preds, target_names=class_names, output_dict=True)
            report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}}) 
            
            concur_report = classification_report(retain_preds, unlearn_preds, target_names=class_names, output_dict=True)
            concur_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": concur_report["accuracy"], "support": concur_report['macro avg']['support']}}) 
            
            report_df = pandas.DataFrame(report).transpose()
            concur_report_df = pandas.DataFrame(concur_report).transpose()
            
            print(report_df)
            print(concur_report_df)
            
            wandb.log({'trace/depoch': (epoch+1)//10, 'trace/accuracy': accuracy, 'trace/retain_model_concurrence': concurrence, 'trace/accuracy_report': wandb.Table(dataframe= report_df), 'trace/concurrency_report': wandb.Table(dataframe = concur_report_df)})
    wandb.finish()
