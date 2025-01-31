import os
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

class dmbioProtDataSetParams():
    def __init__(self, indir, in_file, label_dir, label_fileExt, node_feat_dir, node_feat_fileExt, edge_dir, edge_fileExt, node_cord_dir, node_cord_file_ext):
        self.indir = indir
        self.in_file = in_file
        self.label_dir = label_dir
        self.label_fileExt = label_fileExt
        self.node_feat_dir = node_feat_dir
        self.node_feat_fileExt = node_feat_fileExt
        self.edge_dir = edge_dir
        self.edge_fileExt = edge_fileExt
        self.node_cord_dir = node_cord_dir
        self.node_cord_fileExt = node_cord_file_ext

class dmbioProtDataSet():
    def __init__(self, dsParams):
        #Set parameter for building dataset
        self.dataSetParams = dsParams
        #Load target list (list of Protien's)
        self.targets, self.dataset = self.load_targets()

    def graphcount(self):
        return len(self.targets)
    
    def getgraphbyid(self, idx):
        #Load ondemand
        if self.dataset[idx] is None:
            protId = self.targets[idx]
            graph = self.buildData(protId) #Need to implemet load target function
            self.dataset[idx] = graph 
        
        return self.dataset[idx]

    def load_targets(self):
        input_list_path = os.path.join(self.dataSetParams.indir, self.dataSetParams.in_file)
        targets = []
        dataset = []
        try:
            if not Path(input_list_path).exists():
                raise FileNotFoundError(f"{self.dataSetParams.in_file} not found at {input_list_path}")
            with open(input_list_path, 'r') as flines:
                for line in flines:
                    tgt_name = line.split('.')[0].strip()
                    targets.append(tgt_name)
                    dataset.append(None)
        except Exception as e:
            raise Exception(f'Error loading target names from file {input_list_path}: {e}')
        return targets, dataset
    
    def buildData(self,protId):
        label_file_path = os.path.join(self.dataSetParams.indir, self.dataSetParams.label_dir, f'{protId}.{self.dataSetParams.label_fileExt}')
        node_feat_file_path = os.path.join(self.dataSetParams.indir, self.dataSetParams.node_feat_dir, f'{protId}.{self.dataSetParams.node_feat_fileExt}')
        edge_file_path = os.path.join(self.dataSetParams.indir, self.dataSetParams.edge_dir, f'{protId}.{self.dataSetParams.edge_fileExt}')
        node_cord_file_path = os.path.join(self.dataSetParams.indir, self.dataSetParams.node_cord_dir, f'{protId}.{self.dataSetParams.node_cord_fileExt}')

        if not Path(label_file_path).exists():
                raise FileNotFoundError(f"{protId}.{self.dataSetParams.label_fileExt} not found at {self.dataSetParams.label_dir}")
        if not Path(node_feat_file_path).exists():
                raise FileNotFoundError(f"{protId}.{self.dataSetParams.node_feat_fileExt} not found at {self.dataSetParams.node_feat_dir}")
        if not Path(edge_file_path).exists():
                raise FileNotFoundError(f"{protId}.{self.dataSetParams.edge_fileExt} not found at {self.dataSetParams.edge_dir}")
        if not Path(node_cord_file_path).exists():
                raise FileNotFoundError(f"{protId}.{self.dataSetParams.node_cord_fileExt} not found at {self.dataSetParams.node_cord_dir}")

        #create labels
        # Load the file
        with open(label_file_path, "r") as label_file:
            label_binary_string = label_file.readline().strip()
        # Convert to a list of integers
        label_binary_list = [int(bit) for bit in label_binary_string]
        # Convert to a PyTorch tensor
        label_tensor = torch.tensor(label_binary_list, dtype=torch.float32)
        label_tensor = label_tensor.unsqueeze(dim=1)

        #create node features
        featfile = np.load(node_feat_file_path)
        node_features = torch.Tensor(featfile)

        #### Create edge ####
       # Read the file, skipping the first row
        with open(edge_file_path, "r") as file:
            edgelines = file.readlines()[1:]

        if not edgelines:
            raise Exception(f'No edges found {edge_file_path}')

        # Extract and process edges
        edges = np.array([
            (int(line.split()[0]) - 1, int(line.split()[1]) - 1, float(line.split()[4])) 
            for line in edgelines
        ])

        # Filter valid edges
        valid_mask = (edges[:, 0] < len(node_features)) & (edges[:, 1] < len(node_features))
        
        edges = edges[valid_mask]
        
        if len(edges) == 0:
            raise Exception(f'No valid edges found in {edge_file_path}')

        # Compute weights efficiently
        weights = np.log(np.abs(edges[:, 0] - edges[:, 1])) / edges[:, 2]
        
        # Making edges bi-directional
        src = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
        dst = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
        weights = np.concatenate([weights, weights])

        # Convert to PyTorch tensors
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

        #Cordinate features
        with open(node_cord_file_path, "r") as cord_file:
            cord_flines = cord_file.readlines()

        # Initialize coordinate array
        cord_ca = np.zeros((len(node_features), 3), dtype=np.float32)

        # Extract CA atom coordinates
        ca_atoms = [
            (
                int(line[22:26].strip()) - 1,  # Residue index (zero-based)
                float(line[30:38].strip()),  # X coordinate
                float(line[38:46].strip()),  # Y coordinate
                float(line[46:54].strip())   # Z coordinate
            )
            for line in cord_flines if line.startswith("ATOM") and line[12:16].strip() == "CA"
        ]

        # Assign coordinates to xyz_ca array
        for res_no, x, y, z in ca_atoms:
            if 0 <= res_no < len(cord_ca):
                cord_ca[res_no] = [x, y, z]

        # Convert to PyTorch tensor
        cord_feats = torch.tensor(cord_ca, dtype=torch.float32)

        data = Data(
            x = node_features,
            edge_index = edge_index,
            edge_attr = edge_weights,
            y = label_tensor,
            pos = cord_feats   
        )
        
        return data
    

if __name__ == "__main__":
    #Test case
    import configparser
    #Create your own config with key value pair and change the path of ini file
    config_path = '/teamspace/studios/this_studio/DeepDive/Prot-DNA_experiments/test/testDatasetConfig.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    config_section = config['DEFAULT']

    for key in config_section:
        print(f'{key}: {config_section[key]}')

    dsParams = dmbioProtDataSetParams(
        indir = config_section['indir'],
        in_file = config_section['in_file'],
        label_dir = config_section['label_dir'],
        label_fileExt = config_section['label_fileExt'],
        node_feat_dir = config_section['node_feat_dir'],
        node_feat_fileExt = config_section['node_feat_fileExt'],
        edge_dir = config_section['edge_dir'],
        edge_fileExt = config_section['edge_fileExt'],
        node_cord_dir = config_section['node_cord_dir'],
        node_cord_file_ext = config_section['node_cord_file_ext']
    )
    datasset = dmbioProtDataSet(dsParams)

    print('\n\nBuilding one graph...')
    print(datasset.getgraphbyid(0))
    print('\n\nRetreiving the same graph...')
    print(datasset.getgraphbyid(0))