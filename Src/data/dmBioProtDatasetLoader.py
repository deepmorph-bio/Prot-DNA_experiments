from Src.data.dmbioProtDataset import dmbioProtDataSet, dmbioProtDataSetParams
import os
import configparser
from torch_geometric.loader import  DataLoader

class dmbioProtDatasetloader():
    def __init__(self, configFilePath: str):
        self.configFilePath = configFilePath
        if not os.path.isfile(self.configFilePath):
            raise FileNotFoundError(f"Config file not found at {self.configFilePath}")
        self.dmdataset, self.num_features = self.load_dataset()
        
    def load_dataset(self):
        config = configparser.ConfigParser()
        config.read(self.configFilePath)
        config_section = config['DEFAULT']
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
        req_num_features = config_section.getint('num_features')
        dataset = dmbioProtDataSet(dsParams, req_num_features)
        print("Loading Dataset!")
        # Run load all in case it was not explicitly called
        dataset.load_all()
        return dataset, int(req_num_features)

    def split_train_test_validation(self, train_ratio=0.7, validation_ratio=0.1, batch_size=1):
        #Filter out if num features is not equalto the number of features of the dataset
        dataset_filtered = [graph for graph in self.dmdataset.dataset if graph.num_features == self.num_features and graph.num_nodes == graph.y.shape[0] and graph.y.shape[1] == 1]

        # Split the data into training , validation and test
        num_samples = len(dataset_filtered)
        training = round(num_samples * train_ratio) 
        validation = round(num_samples * validation_ratio)
        test = num_samples - training - validation

        training_sets = dataset_filtered[:training]
        validation_sets = dataset_filtered[training: training+validation]
        test_sets = dataset_filtered[training+validation:]
        #prepare batches 
        batch_training = DataLoader(training_sets, batch_size= batch_size)
        batch_validation = DataLoader(validation_sets, batch_size= batch_size)
        batch_test = DataLoader(test_sets, batch_size= batch_size)

        return batch_training, batch_validation, batch_test
