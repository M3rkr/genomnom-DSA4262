import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Function to plot training progress based on metrics saved during training
def plot_training_progress(model_dir, graph_dir):
    # Define the path to the training log file
    training_log_path = os.path.join(model_dir, 'training_log.json')
    
    # Check if the training log file exists
    if not os.path.exists(training_log_path):
        print("Training log not found.")  # Notify if log is missing
        return  # Exit function if log is not found
    
    # Load the training log data from JSON file
    with open(training_log_path, 'r') as f:
        logs = json.load(f)  # Load JSON data as a Python dictionary
    
    # Generate a list of epochs based on the length of the log entries
    epochs = list(range(1, len(logs['roc_auc']) + 1))
    
    # Create a figure to plot the metrics
    plt.figure(figsize=(10, 6))
    
    # Plot ROC-AUC values for each epoch
    plt.plot(epochs, logs['roc_auc'], label='ROC-AUC')
    
    # Plot PR-AUC values for each epoch
    plt.plot(epochs, logs['pr_auc'], label='PR-AUC')
    
    # Label the x-axis as 'Epoch'
    plt.xlabel('Epoch')
    
    # Label the y-axis as 'Score'
    plt.ylabel('Score')
    
    # Title the plot as 'Training Progress'
    plt.title('Training Progress')
    
    # Display a legend to differentiate between ROC-AUC and PR-AUC lines
    plt.legend()
    
    # Adjust layout for better fit
    plt.tight_layout()
    
    # Save the plot to the specified directory as 'training_progress.png'
    plt.savefig(os.path.join(graph_dir, 'training_progress.png'))
    
    # Close the plot to free up memory
    plt.close()
