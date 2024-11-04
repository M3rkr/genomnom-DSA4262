import os
import json
import itertools
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Function to preprocess data by parsing JSON files, encoding sequences, scaling features, and aggregating
def preprocess_data(data_dir, model_dir):
    # Define file paths for signal data and labels
    signal_data_path = os.path.join(data_dir, 'training', 'dataset0.json')
    labels_data_path = os.path.join(data_dir, 'training', 'data.info.labelled')
    graph_dir = 'graphs'
    os.makedirs(graph_dir, exist_ok=True)  # Create directory for graphs if it doesn't exist
    
    # Load labels data into DataFrame
    labels_df = pd.read_csv(labels_data_path)
    signal_data_list = []  # List to hold processed signal data
    
    # Function to process each transcript, extracting relevant sequence and feature information
    def process_transcript(transcript_id, position_data):
        for position, reads in position_data.items():
            for kmer, features_list in reads.items():
                # Extract sequence windows and feature data
                sequence_pre = kmer[:5]
                sequence_cur = kmer[1:6]
                sequence_post = kmer[2:7]
                # Append data for each feature set within the kmer window
                for features in features_list:
                    signal_data_list.append([transcript_id, position, sequence_pre, sequence_cur, sequence_post] + features)
    
    # Read and process each line from the JSON file (line by line processing)
    with open(signal_data_path, 'r') as file:
        for line in tqdm(file, desc="Processing JSON lines"):
            try:
                data = json.loads(line.strip())
                # Process each transcript and its positions
                for transcript_id, position_data in data.items():
                    process_transcript(transcript_id, position_data)
            except json.JSONDecodeError:
                continue  # Skip any malformed JSON lines
    
    # Define column names for signal data
    signal_columns = [
        'Transcript_ID', 'Position', '5mer_pre', '5mer_cur', '5mer_post',
        'Dwelling_Time_pre', 'Std_Dev_pre', 'Mean_pre',
        'Dwelling_Time_cur', 'Std_Dev_cur', 'Mean_cur',
        'Dwelling_Time_post', 'Std_Dev_post', 'Mean_post'
    ]
    # Convert processed data into a DataFrame
    signal_df = pd.DataFrame(signal_data_list, columns=signal_columns)
    signal_df['Position'] = signal_df['Position'].astype(int)
    
    # Merge signal data with labels on Transcript_ID and Position
    merged_df = pd.merge(
        signal_df,
        labels_df,
        how='inner',
        left_on=['Transcript_ID', 'Position'],
        right_on=['transcript_id', 'transcript_position']
    ).drop(columns=['transcript_id', 'transcript_position'])
    
    # Generate all possible 5-mer sequences and encode them
    bases = ['A', 'C', 'G', 'T']
    all_5mers = [''.join(p) for p in itertools.product(bases, repeat=5)]
    encoder = LabelEncoder()
    encoder.fit(all_5mers)
    joblib.dump(encoder, os.path.join(model_dir, 'label_encoder_5mer.pkl'))  # Save encoder for reuse
    
    # Encode sequence data
    merged_df['5mer_pre_encoded'] = encoder.transform(merged_df['5mer_pre'])
    merged_df['5mer_cur_encoded'] = encoder.transform(merged_df['5mer_cur'])
    merged_df['5mer_post_encoded'] = encoder.transform(merged_df['5mer_post'])
    # Drop original sequence columns after encoding
    merged_df = merged_df.drop(columns=['5mer_pre', '5mer_cur', '5mer_post'])
    
    # Define numerical columns to scale
    numerical_columns = [
        'Dwelling_Time_pre', 'Std_Dev_pre', 'Mean_pre', 
        'Dwelling_Time_cur', 'Std_Dev_cur', 'Mean_cur', 
        'Dwelling_Time_post', 'Std_Dev_post', 'Mean_post'
    ]
    # Scale numerical features to range [0, 1]
    scaler = MinMaxScaler()
    merged_df[numerical_columns] = scaler.fit_transform(merged_df[numerical_columns])
    joblib.dump(scaler, os.path.join(model_dir, 'minmax_scaler.pkl'))  # Save scaler for reuse
    
    # Count the number of reads for each (Transcript_ID, Position) pair
    merged_df['Count'] = merged_df.groupby(['Transcript_ID', 'Position'])['Position'].transform('count')
    # Filter out positions with fewer than 5 reads
    merged_df = merged_df[merged_df['Count'] >= 5]
    
    # Define aggregation functions for each feature
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
    # Aggregate features by (Transcript_ID, Position) pair using defined functions
    aggregated_df = merged_df.groupby(['Transcript_ID', 'Position']).agg(agg_funcs)
    # Flatten multi-level column names created by aggregation
    aggregated_df.columns = ['_'.join(col) for col in aggregated_df.columns]
    aggregated_df = aggregated_df.reset_index()
    
    # Add label column based on the first value in each group
    labels_grouped = merged_df.groupby(['Transcript_ID', 'Position'])['label'].first().reset_index()
    aggregated_df = pd.merge(aggregated_df, labels_grouped, on=['Transcript_ID', 'Position'], how='left')
    
    # If gene_id column is available, group and merge it; otherwise, use Transcript_ID as gene_id
    if 'gene_id' in merged_df.columns:
        genes_grouped = merged_df.groupby(['Transcript_ID', 'Position'])['gene_id'].first().reset_index()
        aggregated_df = pd.merge(aggregated_df, genes_grouped, on=['Transcript_ID', 'Position'], how='left')
    else:
        aggregated_df['gene_id'] = aggregated_df['Transcript_ID']
    
    # Split data into training and testing sets based on genes
    unique_genes = aggregated_df['gene_id'].unique()
    np.random.seed(42)
    test_genes = np.random.choice(unique_genes, size=int(0.2 * len(unique_genes)), replace=False)
    aggregated_df['set_type'] = aggregated_df['gene_id'].apply(lambda gene: 'test' if gene in test_genes else 'train')
    
    # Save the processed and aggregated data to a CSV file
    aggregated_df.to_csv(os.path.join(model_dir, 'aggregated_data.csv'), index=False)
