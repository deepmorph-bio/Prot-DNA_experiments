from torch_geometric.data import Data
from dmbioProtDataset import dmbioProtDataSet, dmbioProtDataSetParams

class dmbioProtDatasetloader():
    def __init__(self, params: dmbioProtDataSetParams):
        self.dsParams = params