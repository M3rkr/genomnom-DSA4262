import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

def select_features(model_dir, graph_dir):
    aggregated_df = pd.read_csv(os.path.join(model_dir, 'aggregated_data.csv'))
    X = aggregated_df.drop(columns=['Transcript_ID', 'Position', 'gene_id', 'label', 'set_type'])
    y = aggregated_df['label']
    X_train = X[aggregated_df['set_type'] == 'train']
    y_train = y[aggregated_df['set_type'] == 'train']

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    selector = SelectFromModel(rf, threshold='median', prefit=True)
    selected_features = X_train.columns[selector.get_support()]
    joblib.dump(selector, os.path.join(model_dir, 'feature_selector.pkl'))

    
    feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances_sorted[:20], y=feature_importances_sorted.index[:20])
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'feature_importances.png'))
    plt.close()
    
    aggregated_df = aggregated_df[['Transcript_ID', 'Position'] + list(selected_features) + ['label', 'set_type', 'gene_id']]
    aggregated_df.to_csv(os.path.join(model_dir, 'aggregated_data_selected.csv'), index=False)
    
    return selected_features