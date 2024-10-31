import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_progress(model_dir, graph_dir):
    training_log_path = os.path.join(model_dir, 'training_log.json')
    if not os.path.exists(training_log_path):
        print("Training log not found.")
        return
    
    with open(training_log_path, 'r') as f:
        logs = json.load(f)
    
    epochs = list(range(1, len(logs['roc_auc']) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, logs['roc_auc'], label='ROC-AUC')
    plt.plot(epochs, logs['pr_auc'], label='PR-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'training_progress.png'))
    plt.close()