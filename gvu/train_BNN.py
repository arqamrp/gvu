import torch
import wandb
from tqdm import tqdm
import pandas
from sklearn.metrics import classification_report, accuracy_score

from sample_model_losses import sample_model_losses

def train_BNN(model, optimizer, train_loader, val_loader, train_config, path, verbose=False):
    num_batches = len(train_loader)
    train_set_size = len(train_loader.dataset)
    val_set_size = len(val_loader.dataset)
    batch_size = train_config['batch_size']
    samples = train_config['sample_size']
    n_epochs = train_config['train_epochs']
    device = train_config['device']
    
    model.train()
    if verbose:
        print(f'Training model on {train_set_size} samples')
    
    # Define custom metrics to ensure correct step tracking
    wandb.define_metric("val/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")
    
    wandb.define_metric("trace/depoch")
    wandb.define_metric("trace/*", step_metric="trace/depoch")
    
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    
    for epoch in tqdm(range(0, n_epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            
            results = sample_model_losses(model, data, target, samples=samples, device=device)
            gvo = results['prior_regularisation_term'] / train_set_size + results['negative_log_likelihood']
            train_nll = results['negative_log_likelihood']
            results['generalized_variational_objective'] = gvo
            
            if verbose:
                print(results)
            
            gvo.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # check if any gradients are nan
            if any(torch.isnan(p.grad if p.grad is not None else torch.zeros(1)).any() for p in model.parameters()):
                print('NaN gradients detected. Skipping update.')
                continue
            else:
                optimizer.step()
            
            # Log to wandb without using `global_step`
            wandb.log({'train/global_step': epoch * num_batches + batch_idx, 
                       **{'train/'+ key: results[key].item() for key in results if results[key] is not None}} )
            
        model.eval()

        val_nll = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                val_results = sample_model_losses(model, data, target, samples=samples, device=device, unlearn=False, div_type=train_config['div_type'], alpha = train_config['alpha'])
                val_nll += val_results['negative_log_likelihood'].item()

        val_nll = val_nll / len(val_loader)
        
        # Log validation NLL using a separate step metric
        wandb.log({'val/epoch': epoch, 'val/nll': val_nll})
        
        # Log histograms of the model weights and biases
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
            print({key: results[key].item() for key in results if results[key] is not None })
            print(f'Epoch {epoch} validation NLL: {val_nll}')
            
            # Evaluation on the test set
            model.eval()
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            # Compute overall accuracy
            accuracy = accuracy_score(all_labels, all_preds)
            print(f'Overall Accuracy: {accuracy:.4f}')
            wandb.log({'trace/depoch': (epoch+1)//10, 'trace/accuracy': accuracy})

            # Compute precision, recall, and F1 score for each class
            class_names = val_loader.dataset.classes
            report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
            report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}}) 
            report_df = pandas.DataFrame(report).transpose()
            wandb.log({'trace/report': wandb.Table(dataframe = report_df)})
            print(report_df)

    wandb.finish()
