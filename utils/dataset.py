import torch
from torch.utils.data import Dataset
import numpy as np

class AggregatedDataset(Dataset):
    def __init__(self, df, feature_columns, target_column):
        self.X = df[feature_columns].values.astype(np.float32)
        self.y = df[target_column].values.astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class PredictionDataset(Dataset):
    def __init__(self, grouped_data):
        self.X = np.array([item[0] for item in grouped_data]).astype(np.float32)
        self.transcript_ids = [item[1] for item in grouped_data]
        self.positions = [item[2] for item in grouped_data]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.transcript_ids[idx], self.positions[idx]