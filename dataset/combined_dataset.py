import torch
import numpy as np
from torch.utils.data import TensorDataset

class CombDataset(torch.utils.data.Dataset):
    """Dataset wrapper to induce class-imbalance"""

    def __init__(self, clean_dataset, adv_dataset, transform=None, return_idx = False):

        #TODO make code change to ensure each batch gets equal representation of clean and adv samples
        self.clean_dataset = clean_dataset ## Clean Dataset
        self.adv_dataset = adv_dataset ## Clean Dataset
        self.combined_dataset = torch.utils.data.ConcatDataset([clean_dataset, adv_dataset])
        self.combined_labels = np.concatenate(([0]*len(self.clean_dataset), [1]*len(self.adv_dataset)))

        print(f'Clean : {sum(self.combined_labels == 0)} \t|\t Adv : {sum(self.combined_labels == 1)}')
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, i):

        (x,cls_), y = self.combined_dataset[i], self.combined_labels[i] ## Original Sample
        
        if self.return_idx:
            return (x, int(cls_), int(y), int(i))
        else: 
            return (x, int(cls_), int(y))   # image, actual image class in dataset, clean/adv 
   
    def __len__(self):
        return len(self.combined_dataset) 
