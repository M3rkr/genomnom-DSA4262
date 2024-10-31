import os
import sys
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import joblib
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils.dataset import PredictionDataset
from utils.utils import FeedforwardNN

def preprocess_new_data(new_data_path, encoder, scaler, selector):
    signal_data_list = []
    def process_transcript(transcript_id, position_data):
        for position, reads in position_data.items():
            for kmer, features_list in reads.items():
                sequence_pre = kmer[:5]
                sequence_cur = kmer[1:6]
                sequence_post = kmer[2:7]
                for features in features_list:
                    signal_data_list.append([transcript_id, position, sequence_pre, sequence_cur, sequence_post] + features)

    with open(new_data_path, 'r') as file:
        for line in tqdm(file, desc="Processing JSON lines"):
            try:
                data = json.loads(line.strip())
                for transcript_id, position_data in data.items():
                    process_transcript(transcript_id, position_data)
            except json.JSONDecodeError:
                continue

    signal_columns = [
        'Transcript_ID', 'Position', '5mer_pre', '5mer_cur', '5mer_post',
        'Dwelling_Time_pre', 'Std_Dev_pre', 'Mean_pre',
        'Dwelling_Time_cur', 'Std_Dev_cur', 'Mean_cur',
        'Dwelling_Time_post', 'Std_Dev_post', 'Mean_post'
    ]
    signal_df = pd.DataFrame(signal_data_list, columns=signal_columns)
    signal_df['Position'] = signal_df['Position'].astype(int)

    signal_df['5mer_pre_encoded'] = encoder.transform(signal_df['5mer_pre'])
    signal_df['5mer_cur_encoded'] = encoder.transform(signal_df['5mer_cur'])
    signal_df['5mer_post_encoded'] = encoder.transform(signal_df['5mer_post'])
    signal_df = signal_df.drop(columns=['5mer_pre', '5mer_cur', '5mer_post'])

    numerical_columns = [
        'Dwelling_Time_pre', 'Std_Dev_pre', 'Mean_pre', 
        'Dwelling_Time_cur', 'Std_Dev_cur', 'Mean_cur', 
        'Dwelling_Time_post', 'Std_Dev_post', 'Mean_post'
    ]
    signal_df[numerical_columns] = scaler.transform(signal_df[numerical_columns])

    agg_funcs = {
        'Dwelling_Time_pre': ['mean', 'std', 'min', 'max', 'median'],
        'Std_Dev_pre': ['mean', 'std', 'min', 'max', 'median'],
        'Mean_pre': ['mean', 'std', 'min', 'max', 'median'],
        'Dwelling_Time_cur': ['mean', 'std', 'min', 'max', 'median'],
        'Std_Dev_cur': ['mean', 'std', 'min', 'max', 'median'],
        'Mean_cur': ['mean', 'std', 'min', 'max', 'median'],
        'Dwelling_Time_post': ['mean', 'std', 'min', 'max', 'median'],
        'Std_Dev_post': ['mean', 'std', 'min', 'max', 'median'],
        'Mean_post': ['mean', 'std', 'min', 'max', 'median']
    }
    aggregated_new_df = signal_df.groupby(['Transcript_ID', 'Position']).agg(agg_funcs)
    aggregated_new_df.columns = ['_'.join(col) for col in aggregated_new_df.columns]
    aggregated_new_df = aggregated_new_df.reset_index()

    X_new = aggregated_new_df.drop(columns=['Transcript_ID', 'Position'])

    X_selected = selector.transform(X_new)

    grouped_data = list(zip(X_selected, aggregated_new_df['Transcript_ID'], aggregated_new_df['Position']))
    return grouped_data


def make_predictions(input_file, output_file, model_dir):
    label_encoder_path = os.path.join(model_dir, 'label_encoder_5mer.pkl')
    scaler_path = os.path.join(model_dir, 'minmax_scaler.pkl')
    selector_path = os.path.join(model_dir, 'feature_selector.pkl')
    best_model_path = os.path.join(model_dir, 'best_aggregated_model.pth')

    encoder = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    best_hyperparams_path = os.path.join(model_dir, 'best_hyperparams.json')
    with open(best_hyperparams_path, 'r') as f:
        best_hyperparams = json.load(f)

    input_dim = sum(selector.get_support())

    model = FeedforwardNN(
        input_dim=input_dim,
        num_layers=best_hyperparams.get('num_layers', 3),
        hidden_size=best_hyperparams.get('hidden_size', 79),
        dropout_rate=best_hyperparams.get('dropout_rate', 0.3087227950951695)
    )
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.eval()

    grouped_data = preprocess_new_data(input_file, encoder, scaler, selector)
    prediction_dataset = PredictionDataset(grouped_data)
    prediction_loader = DataLoader(
        prediction_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0
    )

    all_transcript_ids = []
    all_positions = []
    all_scores = []

    for inputs, transcript_ids, positions in tqdm(prediction_loader, desc="Predicting"):
        outputs = model(inputs).squeeze()
        scores = outputs.detach().cpu().numpy()
        all_transcript_ids.extend(transcript_ids)
        all_positions.extend(positions.tolist())
        all_scores.extend(scores)

    results_df = pd.DataFrame({
        'transcript_id': all_transcript_ids,
        'transcript_position': all_positions,
        'score': all_scores
    })
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def batch_make_predictions(model, model_dir, data_dir, graph_dir):
    input_dir = os.path.join(data_dir, 'prediction')
    output_dir = os.path.join(data_dir, 'prediction')
    os.makedirs(output_dir, exist_ok=True)

    prediction_files = glob.glob(os.path.join(input_dir, '*.json'))

    if not prediction_files:
        print(f"No JSON files found in {input_dir}.")
        return

    print(f"Found {len(prediction_files)} prediction files in {input_dir}.\n")

    for input_path in tqdm(prediction_files, desc="Batch Processing Predictions"):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = f"{base_name}_prediction.csv"
        output_path = os.path.join(output_dir, output_file)
        make_predictions(input_path, output_path, model_dir)

    print("\nBatch prediction completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference Script for m6A Label Prediction')
    parser.add_argument('--input_file', type=str, help='Path to the input JSON file for inference')
    parser.add_argument('--output_file', type=str, help='Path to save the output CSV predictions')
    args = parser.parse_args()

    model_dir = 'model'
    data_dir = 'data'
    graph_dir = 'graphs'

    if args.input_file and args.output_file:
        make_predictions(args.input_file, args.output_file, model_dir)
    else:
        # Load best hyperparameters to configure the model correctly
        best_hyperparams_path = os.path.join(model_dir, 'best_hyperparams.json')
        with open(best_hyperparams_path, 'r') as f:
            best_hyperparams = json.load(f)
        
        # Calculate input_dim correctly
        selector_path = os.path.join(model_dir, 'feature_selector.pkl')
        selector = joblib.load(selector_path)
        input_dim = sum(selector.get_support())

        # Initialize the model with best hyperparameters
        model = FeedforwardNN(
            input_dim=input_dim,
            num_layers=best_hyperparams.get('num_layers', 3),
            hidden_size=best_hyperparams.get('hidden_size', 79),
            dropout_rate=best_hyperparams.get('dropout_rate', 0.3087227950951695)
        )
        best_model_path = os.path.join(model_dir, 'best_aggregated_model.pth')
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        model.eval()
        batch_make_predictions(model, model_dir, data_dir, graph_dir)
