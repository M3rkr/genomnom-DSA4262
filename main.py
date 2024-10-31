import os
import json
from scripts.data_preprocessing import preprocess_data
from scripts.feature_selection import select_features
from scripts.model_training import optimize_hyperparameters, train_final_model
from scripts.evaluation import evaluate_model
from scripts.inference import batch_make_predictions, make_predictions
from scripts.visualization import plot_training_progress

def run_pipeline():
    data_dir = 'data'
    model_dir = 'model'
    graph_dir = 'graphs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    
    preprocess_data(data_dir, model_dir)
    selected_features = select_features(model_dir, graph_dir)
    best_hyperparams = optimize_hyperparameters(model_dir)
    model = train_final_model(best_hyperparams, model_dir)
    evaluate_model(model, model_dir, graph_dir)
    plot_training_progress(model_dir, graph_dir)
    batch_make_predictions(model, model_dir, data_dir, graph_dir)

if __name__ == "__main__":
    run_pipeline()