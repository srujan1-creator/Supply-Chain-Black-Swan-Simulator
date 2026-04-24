import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class M5SyntheticDataset(Dataset):
    """
    A PyTorch Dataset that generates synthetic supply chain demand data.
    Now upgraded to include 'Conditions' (0: Normal, 1: Port Closure, 2: Demand Spike).
    """
    def __init__(self, num_samples=1000, sequence_length=28):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.data, self.conditions = self._generate_data()

    def _generate_data(self):
        # Base demand and seasonality
        base_demand = np.random.normal(loc=10.0, scale=3.0, size=(self.num_samples, self.sequence_length))
        time = np.arange(self.sequence_length)
        seasonality = 5.0 * np.sin(2 * np.pi * time / 7)
        data = base_demand + seasonality
        
        # Generate random conditions: 0 (Normal), 1 (Port Closure), 2 (Spike)
        conditions = np.random.randint(0, 3, size=(self.num_samples,))
        
        # Apply disruptions based on conditions
        for i in range(self.num_samples):
            if conditions[i] == 1:
                # Port Closure: Demand drops to zero for days 10-17
                data[i, 10:17] = np.random.normal(loc=1.0, scale=0.5, size=7)
            elif conditions[i] == 2:
                # Demand Spike: Massive surge on days 20-23 (e.g. viral trend)
                data[i, 20:23] = np.random.normal(loc=30.0, scale=5.0, size=3)
                
        data = np.maximum(data, 0.0) 
        
        # Normalize
        max_val = np.max(data, axis=1, keepdims=True)
        max_val[max_val == 0] = 1.0 
        data = data / max_val
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(conditions, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.conditions[idx]

def get_dataloader(batch_size=32, num_samples=1000, sequence_length=28):
    dataset = M5SyntheticDataset(num_samples, sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dl = get_dataloader(batch_size=4)
    batch_data, batch_cond = next(iter(dl))
    print(f"Data shape: {batch_data.shape}") 
    print(f"Conditions: {batch_cond}")
