import os
import json
import itertools
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_data(data_dir, model_dir):
    signal_data_path = os.path.join(data_dir, 'training', 'dataset0.json')
    labels_data_path = os.path.join(data_dir, 'training', 'data.info.labelled')
    graph_dir = 'graphs'
    os.makedirs(graph_dir, exist_ok=True)
    
    labels_df = pd.read_csv(labels_data_path)
    signal_data_list = []
    
    def process_transcript(transcript_id, position_data):
        for position, reads in position_data.items():
            for kmer, features_list in reads.items():
                sequence_pre = kmer[:5]
                sequence_cur = kmer[1:6]
                sequence_post = kmer[2:7]
                for features in features_list:
                    signal_data_list.append([transcript_id, position, sequence_pre, sequence_cur, sequence_post] + features)
    
    with open(signal_data_path, 'r') as file:
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
    
    merged_df = pd.merge(
        signal_df,
        labels_df,
        how='inner',
        left_on=['Transcript_ID', 'Position'],
        right_on=['transcript_id', 'transcript_position']
    ).drop(columns=['transcript_id', 'transcript_position'])
    
    bases = ['A', 'C', 'G', 'T']
    all_5mers = [''.join(p) for p in itertools.product(bases, repeat=5)]
    encoder = LabelEncoder()
    encoder.fit(all_5mers)
    joblib.dump(encoder, os.path.join(model_dir, 'label_encoder_5mer.pkl'))
    
    merged_df['5mer_pre_encoded'] = encoder.transform(merged_df['5mer_pre'])
    merged_df['5mer_cur_encoded'] = encoder.transform(merged_df['5mer_cur'])
    merged_df['5mer_post_encoded'] = encoder.transform(merged_df['5mer_post'])
    merged_df = merged_df.drop(columns=['5mer_pre', '5mer_cur', '5mer_post'])
    
    numerical_columns = [
        'Dwelling_Time_pre', 'Std_Dev_pre', 'Mean_pre', 
        'Dwelling_Time_cur', 'Std_Dev_cur', 'Mean_cur', 
        'Dwelling_Time_post', 'Std_Dev_post', 'Mean_post'
    ]
    scaler = MinMaxScaler()
    merged_df[numerical_columns] = scaler.fit_transform(merged_df[numerical_columns])
    joblib.dump(scaler, os.path.join(model_dir, 'minmax_scaler.pkl'))
    
    merged_df['Count'] = merged_df.groupby(['Transcript_ID', 'Position'])['Position'].transform('count')
    merged_df = merged_df[merged_df['Count'] >= 5]
    
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
    aggregated_df = merged_df.groupby(['Transcript_ID', 'Position']).agg(agg_funcs)
    aggregated_df.columns = ['_'.join(col) for col in aggregated_df.columns]
    aggregated_df = aggregated_df.reset_index()
    
    labels_grouped = merged_df.groupby(['Transcript_ID', 'Position'])['label'].first().reset_index()
    aggregated_df = pd.merge(aggregated_df, labels_grouped, on=['Transcript_ID', 'Position'], how='left')
    
    if 'gene_id' in merged_df.columns:
        genes_grouped = merged_df.groupby(['Transcript_ID', 'Position'])['gene_id'].first().reset_index()
        aggregated_df = pd.merge(aggregated_df, genes_grouped, on=['Transcript_ID', 'Position'], how='left')
    else:
        aggregated_df['gene_id'] = aggregated_df['Transcript_ID']
    
    unique_genes = aggregated_df['gene_id'].unique()
    np.random.seed(42)
    test_genes = np.random.choice(unique_genes, size=int(0.2 * len(unique_genes)), replace=False)
    aggregated_df['set_type'] = aggregated_df['gene_id'].apply(lambda gene: 'test' if gene in test_genes else 'train')
    
    aggregated_df.to_csv(os.path.join(model_dir, 'aggregated_data.csv'), index=False)
