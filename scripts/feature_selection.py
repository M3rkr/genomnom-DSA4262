import os 
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

# Function to select important features from the dataset using a Random Forest model
def select_features(model_dir, graph_dir):
    # Load the preprocessed aggregated dataset
    aggregated_df = pd.read_csv(os.path.join(model_dir, 'aggregated_data.csv'))
    
    # Separate features (X) and target variable (y)
    X = aggregated_df.drop(columns=['Transcript_ID', 'Position', 'gene_id', 'label', 'set_type'])
    y = aggregated_df['label']
    
    # Filter the training data based on 'set_type' column
    X_train = X[aggregated_df['set_type'] == 'train']
    y_train = y[aggregated_df['set_type'] == 'train']
    
    # Initialize and train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)  # Train on the training data
    
    # Use feature selection based on feature importances from the Random Forest
    selector = SelectFromModel(rf, threshold='median', prefit=True)  # Prefit to use model directly
    # Get the names of selected features based on the specified threshold (median importance)
    selected_features = X_train.columns[selector.get_support()]
    # Save the feature selector model for future use
    joblib.dump(selector, os.path.join(model_dir, 'feature_selector.pkl'))
    
    # Analyze feature importances and sort them in descending order
    feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)
    
    # Plot and save the top 20 feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances_sorted[:20], y=feature_importances_sorted.index[:20])
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'feature_importances.png'))
    plt.close()  # Close the plot to avoid memory issues
    
    # Create a new DataFrame with only selected features and essential columns
    aggregated_df = aggregated_df[['Transcript_ID', 'Position'] + list(selected_features) + ['label', 'set_type', 'gene_id']]
    
    # Save the filtered dataset with selected features
    aggregated_df.to_csv(os.path.join(model_dir, 'aggregated_data_selected.csv'), index=False)
    
    # Return the names of selected features for further analysis or use
    return selected_features
