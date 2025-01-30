import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class dmbioProtDataSet(Dataset):
    def __init__(self, indir):
        #Set indirectory of dataset
        self.indir = indir
        #Load target list (list of Protien's)
        self.targets = self.load_targets()
        self.data = {}    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        #Load ondemand
        if idx in self.data:
            return self.data[idx].graph , self.data[idx].label
        else:
            #Need to implemet load target function
            return None, None

    def load_targets(self):
        input_list_path = os.path.join(self.indir, 'input.list')
        targets = []
        try:
            if not Path(input_list_path).exists():
                raise FileNotFoundError(f"input.list not found at {input_list_path}")
            with open(input_list_path, 'r') as flines:
                for line in flines:
                    tgt_name = line.split('.')[0].strip()
                    targets.append(tgt_name)
        except Exception as e:
            raise Exception(f'Error loading target names from file {input_list_path}: {e}')
        return targets