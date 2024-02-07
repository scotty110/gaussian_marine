'''
Functions to help train models
'''

import torch
import pandas as pd 
import numpy as np

from torch.utils.data import Dataset, DataLoader

'''
Data Loader
Functions:
    - get_split: Get the indices for the training and testing data
    - MarineDataset: Dataset class for the ncei marine data file that was process by data_explore.ipynb
    - MarineDataLoader: Dataloader class for the ncei marine data file that was process by data_explore.ipynb
    - partition_data: Partition the data into training and testing data
'''
def get_split(size:int, split:float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Split indecies into training and testing
    Inputs:
        - size (int): Size of the dataset
        - split (float): Percentage of the data to be used for testing
    Outputs:
        - train_indices (np.ndarray): Indices for the training data
        - test_indices (np.ndarray): Indices for the testing data
    '''
    # Get the indices for the training and testing data
    indices = list(range(size))
    np.random.shuffle(indices)
    split = int(np.floor(split* size))
    train_indices, test_indices = indices[split:], indices[:split]
    return train_indices, test_indices


class MarineDataset(Dataset):
    def __init__(self, data_path:str, split:np.ndarray):
        '''
        Simple dataset class for the ncei marine data file that was process by data_explore.ipynb
        Inputs:
            - data_path (str): Path to the data file
            - split (np.ndarray): Indices to use for dataset (will be split into train and test)
        '''
        df = pd.read_parquet(data_path)
        df = df.dropna()
        self.data = df.iloc[split]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Get item for iterator, returns X, y pair from row idx
        Inputs:
            - idx (int): Index to get X, y pair from
        Outputs:
            - X (torch.Tensor): Input data
            - y (torch.Tensor): Output data
        '''
        row = self.data.iloc[idx].to_dict()
        X = torch.tensor([row['WindSpeed'], row['WetTemp'], row['SeaTemp'], row['CloudAmount'] ], dtype=torch.float64) # Actually select items TODO
        y = torch.tensor([row['AirTemp']], dtype=torch.float64)
        return X, y


class MarineDataLoader(DataLoader):
    def __init__(self, data_path:str, split:np.ndarray, batch_size:int=256, shuffle:bool=True, num_workers:int=0):
        '''
        Simple dataloader class for the ncei marine data file that was process by data_explore.ipynb
        Inputs:
            - data_path (str): Path to the data file
            - split (np.ndarray): Indices to use for dataset (will be split into train and test)
            - batch_size (int): Batch size for the dataloader
            - shuffle (bool): Whether to shuffle the data
            - num_workers (int): Number of workers to use for the dataloader
        '''
        self.dataset = MarineDataset(data_path, split)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __len__(self) -> int:
        return len(self.dataset) 


def partition_data(data_path:str, split:float=0.3) -> tuple[MarineDataLoader, MarineDataLoader]:
    '''
    Partition the data into training and testing data
    Inputs:
        - data_path (str): Path to the data file
        - split (float): Percentage of the data to be used for testing, default is 0.3
    Outputs:
        - train_loader (MarineDataLoader): Dataloader for the training data
        - test_loader (MarineDataLoader): Dataloader for the testing data
    '''
    # Get the indices for the training and testing data
    df = pd.read_parquet(data_path)
    df = df.dropna() # Drop rows with NaN values, do this in dataloader so... 
    train_indices, test_indices = get_split(len(df), split)

    # Create the dataloaders
    train_loader = MarineDataLoader(data_path, train_indices)
    test_loader = MarineDataLoader(data_path, test_indices)
    return train_loader, test_loader


'''
Want to try down sampling (Latin Hyper Cude, adaptive design) data to see if it helps with training
'''
def partition_df(df:pd.DataFrame, split:float=0.3) -> tuple[MarineDataLoader, MarineDataLoader]:
    '''
    Partition the data into training and testing data
    Inputs:
        - df (pd.DataFrame): Dataframe to use for the train and test dataloaders
    Outputs:
        - train_loader (MarineDataLoader): Dataloader for the training data
        - test_loader (MarineDataLoader): Dataloader for the testing data
    '''
    # Get the indices for the training and testing data
    df = df.dropna() # Drop rows with NaN values, do this in dataloader so... 
    train_indices, test_indices = get_split(len(df), split)

    # Create the split 
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    return train_df, test_df

'''
Model Training
''' 

'''
TODO:
    - Cross validation
    - Plot loss
    - Checkpointing
    - Train Loop
    - Test Loop
    - Step Loop
'''
