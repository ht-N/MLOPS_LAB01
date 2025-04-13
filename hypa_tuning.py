# hyperparameters tuning
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import the functions from train.py
from train import get_dataset, get_model, train_epoch, validate, evaluate, save_checkpoint

# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'batch_size': {'values': [32, 64, 128, 256]},
        'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
        'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-3},
        'optimizer': {'values': ['adam', 'sgd']},
        'model': {'values': ['resnet18', 'resnet50']},
        'epochs': {'value': 10}
    }
}

def train_with_hyperparams():
    wandb.init()
    # Get configuration from sweep
    config = wandb.config
    
    args = argparse.Namespace(
        data_dir='./data',
        dataset_name='CIFAR10',
        dataset_version='1.0',
        batch_size=config.batch_size,
        lr=config.lr,
        epochs=config.epochs,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        model=config.model,
        wandb_project='mlops-hyperparameter-tuning',
        wandb_entity=None,
        exp_name=f"sweep_trial_{config.model}_{config.optimizer}_lr{config.lr}_bs{config.batch_size}",
        no_wandb=False,
        checkpoint_dir='./checkpoints'
    )
    
    # Make sure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # preprocessing
    train_loader, val_loader, test_loader, classes = get_dataset(args)
    
    # Create model
    model = get_model(args.model, num_classes=len(classes))
    model = model.to(device)
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    
    # optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                             weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    
    # Training loop
    best_val_acc = 0.0
    
    # Training and validation
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print progress
        print(f'Epoch {epoch+1}/{args.epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Log metrics to wandb
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        wandb.log(log_dict)
        
        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.exp_name}.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            
            wandb.run.summary["best_val_acc"] = best_val_acc
            wandb.run.summary["best_epoch"] = epoch + 1
    
    # Evaluation on test set
    print("Evaluating model on test set...")
    test_metrics = evaluate(model, test_loader, device, classes)
    
    print("Test metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Log test metrics to wandb
    wandb.log(test_metrics)
    
    # Also add to run summary for better visibility in the UI
    for metric_name, metric_value in test_metrics.items():
        wandb.run.summary[metric_name] = metric_value
    
    # Create a summary table for test metrics
    data = [[metric_name, metric_value] for metric_name, metric_value in test_metrics.items()]
    table = wandb.Table(data=data, columns=["Metric", "Value"])
    wandb.log({"test_metrics_table": table})

def main():
    parser = argparse.ArgumentParser(description='Run WandB sweeps for hyperparameter tuning')
    parser.add_argument('--sweep_id', type=str, default=None, 
                       help='Existing sweep ID to use, if not provided a new sweep will be created')
    parser.add_argument('--project', type=str, default='MLOPS_LAB01',
                       help='WandB project name for the sweep')
    parser.add_argument('--entity', type=str, default=None, 
                        help='WandB entity name')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of runs to execute in the sweep')
    
    args = parser.parse_args()
    
    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"Created sweep with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"Using existing sweep with ID: {sweep_id}")
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=train_with_hyperparams, count=args.count)

if __name__ == '__main__':
    main()