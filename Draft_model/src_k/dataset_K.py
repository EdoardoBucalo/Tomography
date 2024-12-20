import lightning as L
import torch
import numpy as np
import os
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from torch.utils.data import Subset
import create_db as c_db
import utils

# Togliere la dataset class e usare direttamente il datamodule?
class TomographyDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, file_name):
    self.data_dir = data_dir
    self.file_name = file_name
    # Load the dataset
    bloated_dataset = np.load(
      os.path.join(self.data_dir, self.file_name),
      allow_pickle=True
      )
    # Here i have the entire dataset in a dictionary with keys 'data' and 'target'
    # Data and target are the input and output of the model
    self.data = torch.Tensor(bloated_dataset['data'])
    self.target = torch.Tensor(bloated_dataset['target'])
    # The following data is not used duing training, but it is useful for visualization
    self.labels = bloated_dataset['label']
    self.shots = bloated_dataset['shot']
    self.time = bloated_dataset['time']
    self.dataerr = bloated_dataset['data_err']
    self.emiss = bloated_dataset['emiss']
    self.x_emiss = bloated_dataset['x_emiss'][0]
    self.y_emiss = bloated_dataset['y_emiss'][0]
    self.majr = bloated_dataset['majr'][0]
    self.minr = bloated_dataset['minr'][0]
    self.b_tor = bloated_dataset['b_tor']
    self.b_rad = bloated_dataset['b_rad']
    self.phi_tor = bloated_dataset['phi_tor']
    self.j0, self.j1, self.em, self.em_hat, self.radii, self.angles = utils.compute_bessel_n_mesh(self.minr, self.majr, self.x_emiss, self.y_emiss)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.target[idx]#, self.j0, self.j1, self.em, self.em_hat, self.radii, self.angles

class TomographyDataModule(L.LightningDataModule):
    def __init__(self, data_dir, file_name, batch_size, num_workers=4,k_folds=5):
        super().__init__()
        self.data_dir = data_dir
        self.file_name = file_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_folds = k_folds
        self.folds = None

    def prepare_data(self):
        c_db.create_db()

    def setup(self, stage=None, fold_idx=None):
        '''
        The setup method is called before the dataloaders are created. Its main 
        purpose is to:
        - Load the dataset from the data directory.
        - Specifically load only the brilliance and 
        - split the dataset into training, validation, and test sets.
        '''
        # Load the dataset
        entire_dataset = TomographyDataset(self.data_dir, self.file_name)
        
        # Normalize the data but only the ones that are greater than 0
        mask_pos = entire_dataset.data > 0  # Get the data containing real values from the diagnostic
        mask_neg = entire_dataset.data < 0  # Get the data not containing real values from the diagnostic
        entire_dataset.data[mask_pos] = (entire_dataset.data[mask_pos] - entire_dataset.data[mask_pos].mean()) / entire_dataset.data[mask_pos].std()  # Normalize the real values
        entire_dataset.data[mask_neg] = -10  # Set the non-real values to -10
        
        # Normalize the targets
        self.mean = entire_dataset.target.mean()
        self.std = entire_dataset.target.std()
        entire_dataset.target = (entire_dataset.target - self.mean) / self.std
        
        # Split the dataset into train, validation, and test sets using K-fold
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        self.folds = list(kfold.split(entire_dataset.data))  # Indices for each fold
        
        # Handling 'fit' or 'test' based on 'stage' or 'fold_idx'
        if stage == 'fit' and fold_idx is not None:
            # Set up the train and validation subsets based on the fold index for training
            train_indices, val_indices = self.folds[fold_idx]
            self.train_subset = Subset(entire_dataset, train_indices)
            self.val_subset = Subset(entire_dataset, val_indices)
            self.train_ds = self.train_subset
            self.val_ds = self.val_subset
        
        elif stage == 'test':
            # Use the entire dataset for testing (or implement separate test set if needed)
            self.test_ds = entire_dataset
        
        else:
            # Default case if neither stage is provided or we're not in a specific fold
            self.train_ds = entire_dataset  # Full dataset for training
            self.val_ds = entire_dataset  # Full dataset for validation (adjust as needed)
    
 

    def train_dataloader(self):
        # Return the train dataloader
        return DataLoader(self.train_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      shuffle=True,
                      pin_memory=True)

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(self.val_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      shuffle=False,
                      pin_memory=True)

    #def test_dataloader(self):
        # Return the test dataloader
        #return DataLoader(self.test_ds,
                      #batch_size=self.batch_size,
                      #num_workers=self.num_workers,
                      #shuffle=False,
                      #pin_memory=True)
  
