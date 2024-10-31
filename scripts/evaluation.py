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

def evaluate_model(model, model_dir, graph_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregated_df = pd.read_csv(os.path.join(model_dir, 'aggregated_data_selected.csv'))
    feature_columns = [col for col in aggregated_df.columns if col not in ['Transcript_ID', 'Position', 'gene_id', 'label', 'set_type']]
    target_column = 'label'
    test_df = aggregated_df[aggregated_df['set_type'] == 'test']
    test_dataset = AggregatedDataset(test_df, feature_columns, target_column)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    try:
        final_roc_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        final_roc_auc = 0.0
    try:
        final_pr_auc = average_precision_score(all_labels, all_preds)
    except ValueError:
        final_pr_auc = 0.0
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'roc_curve.png'))
    plt.close()
    
    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    pr_auc_val = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc_val:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'pr_curve.png'))
    plt.close()
    
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    predicted_classes = (np.array(all_preds) >= best_threshold).astype(int)
    report = classification_report(all_labels, predicted_classes)
    print(report)
    
    return all_preds, all_labels, best_threshold