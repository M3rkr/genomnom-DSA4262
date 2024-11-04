import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
import optuna
from imblearn.over_sampling import SMOTE
from utils.dataset import AggregatedDataset
from utils.utils import FeedforwardNN, set_seed
from tqdm import tqdm

# Function to optimize model hyperparameters using Optuna
def optimize_hyperparameters(model_dir):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the preprocessed dataset
    aggregated_df = pd.read_csv(os.path.join(model_dir, 'aggregated_data_selected.csv'))
    
    # Identify feature columns and target column
    feature_columns = [col for col in aggregated_df.columns if col not in ['Transcript_ID', 'Position', 'gene_id', 'label', 'set_type']]
    target_column = 'label'
    
    # Split the dataset into training and testing sets based on 'set_type' column
    train_df = aggregated_df[aggregated_df['set_type'] == 'train']
    test_df = aggregated_df[aggregated_df['set_type'] == 'test']
    
    # Apply SMOTE to handle class imbalance in the training set
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(train_df[feature_columns], train_df[target_column])
    
    # Create a new training DataFrame with SMOTE-applied data
    train_smote_df = pd.DataFrame(X_train_smote, columns=feature_columns)
    train_smote_df[target_column] = y_train_smote
    
    # Convert the training and testing data into PyTorch datasets
    train_dataset = AggregatedDataset(train_smote_df, feature_columns, target_column)
    test_dataset = AggregatedDataset(test_df, feature_columns, target_column)

    # Define the objective function for Optuna hyperparameter tuning
    def objective(trial):
        # Define the hyperparameter search space
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 5)
        hidden_size = trial.suggest_int('hidden_size', 64, 512)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'ExponentialLR', 'None'])
        
        # Initialize the neural network model
        model = FeedforwardNN(len(feature_columns), num_layers, hidden_size, dropout_rate).to(device)
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

        # Choose the optimizer based on trial suggestion
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Set learning rate scheduler based on trial suggestion
        if scheduler_name == 'StepLR':
            step_size = trial.suggest_int('step_size', 5, 15)
            gamma = trial.suggest_float('gamma', 0.1, 0.5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'ExponentialLR':
            gamma = trial.suggest_float('gamma', 0.9, 0.99)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            scheduler = None

        # DataLoader for the training data
        train_loader_trial = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        num_epochs = 20
        best_val_roc_auc = 0.0  # Track the best validation ROC AUC score
        patience = 5  # Early stopping patience
        epochs_no_improve = 0  # Counter for early stopping

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            # Loop through batches
            for inputs, labels in train_loader_trial:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader_trial.dataset)

            # Validation phase
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for inputs, labels in DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            try:
                val_roc_auc = roc_auc_score(val_labels, val_preds)
            except ValueError:
                val_roc_auc = 0.0

            # Update scheduler if defined
            if scheduler:
                scheduler.step()

            # Early stopping check based on ROC AUC
            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
        return best_val_roc_auc

    # Use Optuna to find the best hyperparameters
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100, timeout=3600)
    
    # Save the best hyperparameters
    best_trial = study.best_trial
    with open(os.path.join(model_dir, 'best_hyperparams.json'), 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    
    return best_trial.params

# Function to train the final model with the best hyperparameters
def train_final_model(best_params, model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    aggregated_df = pd.read_csv(os.path.join(model_dir, 'aggregated_data_selected.csv'))
    feature_columns = [col for col in aggregated_df.columns if col not in ['Transcript_ID', 'Position', 'gene_id', 'label', 'set_type']]
    target_column = 'label'
    train_df = aggregated_df[aggregated_df['set_type'] == 'train']
    test_df = aggregated_df[aggregated_df['set_type'] == 'test']

    # Apply SMOTE to balance classes in the training set
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(train_df[feature_columns], train_df[target_column])
    train_smote_df = pd.DataFrame(X_train_smote, columns=feature_columns)
    train_smote_df[target_column] = y_train_smote

    # Create datasets for training and testing
    train_dataset = AggregatedDataset(train_smote_df, feature_columns, target_column)
    test_dataset = AggregatedDataset(test_df, feature_columns, target_column)
    
    # Initialize DataLoader with best batch size
    batch_size = best_params['batch_size']
    train_loader_final = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize the model with the best hyperparameters
    model = FeedforwardNN(
        input_dim=len(feature_columns),
        num_layers=best_params['num_layers'],
        hidden_size=best_params['hidden_size'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    criterion = nn.BCELoss()  # Loss function for binary classification

    # Initialize optimizer and scheduler based on best parameters
    optimizer_name = best_params['optimizer']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    scheduler_name = best_params.get('scheduler', 'None')
    if scheduler_name == 'StepLR':
        step_size = best_params.get('step_size', 10)
        gamma = best_params.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'ExponentialLR':
        gamma = best_params.get('gamma', 0.95)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Training loop configurations
    num_epochs = 100  # Maximum number of epochs
    patience = 10  # Early stopping patience
    best_pr_auc = 0.0  # Track the best Precision-Recall AUC score
    best_roc_auc = 0.0  # Track the best ROC AUC score
    best_model_state = None  # Save the state of the best model
    epochs_no_improve = 0  # Counter for early stopping
    training_log = {'roc_auc': [], 'pr_auc': []}  # Log training metrics for analysis

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # Track the running loss for each epoch

        # Iterate over batches in the training DataLoader
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader_final, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device (GPU/CPU)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            running_loss += loss.item() * inputs.size(0)  # Accumulate batch loss

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader_final.dataset)

        # Validation phase: evaluate model on test data
        model.eval()  # Set model to evaluation mode
        val_preds = []
        val_labels = []
        
        with torch.no_grad():  # Disable gradient calculation for evaluation
            # Iterate over batches in the test DataLoader
            for inputs, labels in DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)  # Forward pass
                val_preds.extend(outputs.cpu().numpy())  # Collect predictions
                val_labels.extend(labels.cpu().numpy())  # Collect true labels

        # Calculate evaluation metrics
        try:
            val_roc_auc = roc_auc_score(val_labels, val_preds)  # ROC AUC score
        except ValueError:
            val_roc_auc = 0.0  # Handle cases where ROC AUC cannot be calculated
        
        try:
            val_pr_auc = average_precision_score(val_labels, val_preds)  # Precision-Recall AUC
        except ValueError:
            val_pr_auc = 0.0  # Handle cases where PR AUC cannot be calculated

        # Record metrics for analysis
        training_log['roc_auc'].append(val_roc_auc)
        training_log['pr_auc'].append(val_pr_auc)

        # Step the learning rate scheduler if one was defined
        if scheduler:
            scheduler.step()

        # Early stopping and model checkpointing based on PR AUC and ROC AUC
        if val_pr_auc > best_pr_auc or val_roc_auc > best_roc_auc:
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc  # Update best PR AUC
            if val_roc_auc > best_roc_auc:
                best_roc_auc = val_roc_auc  # Update best ROC AUC
            best_model_state = model.state_dict()  # Save model state for best performance
            epochs_no_improve = 0  # Reset early stopping counter
            # Save the best model to disk
            torch.save(best_model_state, os.path.join(model_dir, 'best_aggregated_model.pth'))
        else:
            epochs_no_improve += 1  # Increment early stopping counter if no improvement
            if epochs_no_improve >= patience:  # Trigger early stopping if patience exceeded
                break

    # Save training metrics to a JSON file for later analysis
    with open(os.path.join(model_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f)

    # Load the best model state (if available)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model  # Return the trained model
