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

def parse_args():
    parser = argparse.ArgumentParser(description='Training pipeline with wandb tracking')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('--dataset_version', type=str, default='1.0', help='Dataset version')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer (adam, sgd)')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture')
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default='MLOPS_LAB01', 
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    parser.add_argument('--exp_name', type=str, default='logging_demo', help='Experiment name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help='Directory to save model checkpoints')
    
    return parser.parse_args()

def get_dataset(args):
    """Data preprocessing step: Load and prepare dataset"""
    if args.dataset_name.lower() == 'cifar10':
        # Define data transformations for preprocessing
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load datasets
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, 
                                         download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, 
                                        download=True, transform=transform_test)
        
        # Split training data into train and validation
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=2)
        
        # Classes for logging
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck')
        
        return train_loader, val_loader, test_loader, classes
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported yet")

def get_model(model_name, num_classes=10, pretrained=True):
    """Create a model based on the specified architecture"""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported yet")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    
    return val_loss, val_acc

def evaluate(model, test_loader, device, classes):
    """Evaluate the model and compute metrics"""
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Compute evaluation metrics
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_predictions),
        'precision_macro': precision_score(all_targets, all_predictions, average='macro'),
        'recall_macro': recall_score(all_targets, all_predictions, average='macro'),
        'f1_macro': f1_score(all_targets, all_predictions, average='macro')
    }
    
    # Per-class metrics
    for i, class_name in enumerate(classes):
        class_pred = (all_predictions == i)
        class_true = (all_targets == i)
        metrics[f'precision_{class_name}'] = precision_score(class_true, class_pred, zero_division=0)
        metrics[f'recall_{class_name}'] = recall_score(class_true, class_pred, zero_division=0)
        metrics[f'f1_{class_name}'] = f1_score(class_true, class_pred, zero_division=0)
    
    return metrics

def save_checkpoint(model, optimizer, epoch, val_acc, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, filename)
    
def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=vars(args),
            # Increase timeout if needed
            settings=wandb.Settings(start_method="thread", init_timeout=120)
        )
        
        # Log dataset information
        wandb.config.update({
            "dataset": {
                "name": args.dataset_name,
                "version": args.dataset_version,
                "source": "torchvision"
            }
        })
    
    # preprocessing
    train_loader, val_loader, test_loader, classes = get_dataset(args)
    
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
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
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
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict)
            
            # Log gradients and parameters
            wandb.watch(model, log="all", log_freq=10)
        
        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{args.exp_name}.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            
            if not args.no_wandb:
                wandb.save(checkpoint_path)  # Log checkpoint to wandb
                wandb.run.summary["best_val_acc"] = best_val_acc
                wandb.run.summary["best_epoch"] = epoch + 1
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = os.path.join(args.checkpoint_dir, 
                                          f'model_{args.exp_name}_epoch{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            
            if not args.no_wandb:
                wandb.save(checkpoint_path)
    
    # Evaluation on test set
    print("Evaluating model on test set...")
    test_metrics = evaluate(model, test_loader, device, classes)
    
    print("Test metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Log test metrics to wandb
    if not args.no_wandb:
        # Log metrics as a step
        wandb.log(test_metrics)
        
        # Also add to run summary for better visibility in the UI
        for metric_name, metric_value in test_metrics.items():
            wandb.run.summary[metric_name] = metric_value
        
        # Create a summary table for test metrics
        data = [[metric_name, metric_value] for metric_name, metric_value in test_metrics.items()]
        table = wandb.Table(data=data, columns=["Metric", "Value"])
        wandb.log({"test_metrics_table": table})
        
        # Finish wandb run
        wandb.finish()

if __name__ == '__main__':
    main() 