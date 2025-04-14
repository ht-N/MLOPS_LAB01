# Machine Learning Training Pipeline with Weights & Biases Integration

[Demonstration Video Link](https://youtu.be/3wwte5sSnCM?si=NKCX09EMTSSrlE5V)

## Introduction to the Pipeline

This repository contains a comprehensive machine learning training pipeline built with PyTorch and integrated with Weights & Biases (wandb) for experiment tracking. The pipeline is designed to train image classification models on the CIFAR-10 dataset, with support for various model architectures and hyperparameters. The pipeline includes a standard training workflow (`train.py`) and an automated hyperparameter tuning component (`hypa_tuning.py`) to find optimal model configurations.

Key features of the pipeline:
- Model training with various architectures (ResNet18, ResNet50)
- Hyperparameter optimization with Bayesian search
- Comprehensive metrics tracking and visualization
- Model checkpointing and experiment management
- Detailed performance evaluation

## Pipeline Steps

The pipeline consists of the following key steps:

### 1. Data Preparation
- Loading and preprocessing the CIFAR-10 dataset
- Applying data augmentation techniques (random cropping, horizontal flipping)
- Normalizing pixel values
- Splitting data into training, validation, and test sets

### 2. Model Configuration
- Selection of model architecture (ResNet18, ResNet50)
- Configuration of hyperparameters (learning rate, batch size, optimizer, etc.)
- Initialization of optimizer and loss function

### 3. Training Process
- Iterative training over multiple epochs
- Forward and backward passes
- Gradient-based optimization
- Regular validation to monitor model performance

### 4. Evaluation and Metrics
- Calculation of comprehensive evaluation metrics:
  - Accuracy, precision, recall, F1-score
  - Per-class performance metrics
- Visualization of results through Weights & Biases

### 5. Hyperparameter Tuning (in `hypa_tuning.py`)
- Definition of hyperparameter search space
- Bayesian optimization to find optimal hyperparameters
- Multiple training runs with different configurations
- Comparison of results across configurations

### 6. Model Saving and Deployment
- Saving the best model checkpoints
- Recording model metadata and parameters
- Enabling reproducibility of results

## Weights & Biases Framework

[Weights & Biases (wandb)](https://wandb.ai/) is a powerful experiment tracking tool integrated into our pipeline that offers several key capabilities:

### Key Features
- **Experiment Tracking**: Automatically logs all metrics, hyperparameters, and system information during training
- **Visualization**: Real-time charts and graphs to monitor training progress
- **Hyperparameter Optimization**: Built-in support for hyperparameter sweeps using various optimization strategies
- **Artifact Management**: Version control for datasets and models
- **Collaboration**: Sharing results and insights across teams

### Benefits in this Pipeline
- **Reproducibility**: All experiments are logged with their complete configuration
- **Comparison**: Easy side-by-side comparison of different models and hyperparameters
- **Resource Monitoring**: Tracking of GPU usage, training time, and other system metrics
- **Reports**: Generation of shareable reports with visualizations and insights
- **Sweeps**: Automated hyperparameter tuning with built-in optimization algorithms

Weights & Biases not only helps with tracking experiments but also assists in making informed decisions about model selection and hyperparameter choices.

## Setup and Running the Code

### Setting up the Environment

#### Using Anaconda

```bash
# Create a new conda environment
conda create -n MLOPS_TH01 python=3.11

# Activate the environment
conda activate MLOPS_TH01

# Install PyTorch with CUDA support (adjust according to your CUDA version, mine is 12.6)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
scikit-learn>=0.24.0
wandb>=0.10.0
```

### Running the Training Pipeline

To run the standard training pipeline:

```bash
# Basic usage with default parameters
python train.py

# Customized training with specific parameters
python train.py --model resnet50 --batch_size 128 --epochs 20 --optimizer sgd --lr 0.01

# Training without Weights & Biases logging
python train.py --no_wandb

# Full parameter list
python train.py --help
```

Common parameters:
- `--model`: Model architecture to use (default: resnet18)
- `--batch_size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 10)
- `--optimizer`: Optimization algorithm (default: adam)
- `--wandb_project`: Weights & Biases project name (default: MLOPS_LAB01)
- `--exp_name`: Experiment name for tracking (default: logging_demo)

### Running Hyperparameter Tuning

To run the hyperparameter optimization:

```bash
# Create a new sweep and run 10 experiments
python hypa_tuning.py --count 10

# Use an existing sweep ID and run additional experiments
python hypa_tuning.py --sweep_id <your-sweep-id> --count 5

# Customize the project name
python hypa_tuning.py --project "My_Custom_Project"
```

The hyperparameter tuning will automatically search through combinations of:
- Model architecture (ResNet18, ResNet50)
- Batch size (32, 64, 128, 256)
- Learning rate (log-uniform from 1e-4 to 1e-2)
- Weight decay (log-uniform from 1e-6 to 1e-3)
- Optimizer (Adam, SGD)

All results will be logged to your Weights & Biases account, where you can analyze the performance of different configurations and identify the best hyperparameters for your specific use case.
