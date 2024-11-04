import os
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from utils.dataset import AggregatedDataset
from utils.utils import FeedforwardNN

# Function to evaluate the model on the test dataset and generate performance metrics
def evaluate_model(model, model_dir, graph_dir):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the preprocessed dataset
    aggregated_df = pd.read_csv(os.path.join(model_dir, 'aggregated_data_selected.csv'))
    
    # Define columns used for features, excluding metadata columns
    feature_columns = [col for col in aggregated_df.columns if col not in ['Transcript_ID', 'Position', 'gene_id', 'label', 'set_type']]
    target_column = 'label'  # Target column for prediction
    
    # Filter test set based on 'set_type' column
    test_df = aggregated_df[aggregated_df['set_type'] == 'test']
    
    # Initialize dataset and data loader for the test set
    test_dataset = AggregatedDataset(test_df, feature_columns, target_column)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )
    
    # Set model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()
    all_preds = []  # List to store model predictions
    all_labels = []  # List to store actual labels
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Get model predictions
            all_preds.extend(outputs.cpu().numpy())  # Append predictions to list
            all_labels.extend(labels.cpu().numpy())  # Append labels to list
    
    # Calculate ROC AUC score (try-except to handle cases with no positive samples)
    try:
        final_roc_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        final_roc_auc = 0.0  # Default to 0.0 if calculation fails due to lack of positive samples
    
    # Calculate Precision-Recall AUC score (try-except to handle cases with no positive samples)
    try:
        final_pr_auc = average_precision_score(all_labels, all_preds)
    except ValueError:
        final_pr_auc = 0.0  # Default to 0.0 if calculation fails
    
    # Generate and plot the ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    # Save ROC curve plot to specified directory
    plt.savefig(os.path.join(graph_dir, 'roc_curve.png'))
    plt.close()  # Close the plot
    
    # Generate and plot the Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    pr_auc_val = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc_val:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    # Save Precision-Recall curve plot to specified directory
    plt.savefig(os.path.join(graph_dir, 'pr_curve.png'))
    plt.close()  # Close the plot
    
    # Calculate F1 scores for each threshold in the Precision-Recall curve
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)  # Add small epsilon to avoid division by zero
    # Find the threshold that gives the maximum F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]
    # Convert predictions to binary class labels using the best threshold
    predicted_classes = (np.array(all_preds) >= best_threshold).astype(int)
    # Generate classification report (precision, recall, F1-score)
    report = classification_report(all_labels, predicted_classes)
    print(report)  # Print the classification report
    
    # Return model predictions, actual labels, and the optimal threshold
    return all_preds, all_labels, best_threshold
