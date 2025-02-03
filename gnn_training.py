import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import pytorch_lightning as pl
import Src.model.dmBioGNN as dmBioModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from Src.data.dmbioProtDataset import dmbioProtDataSetParams, dmbioProtDataSet

device_count = torch.cuda.device_count() - 1 if torch.cuda.is_available() else 0
device = torch.device(f'cuda:{device_count}') if torch.cuda.is_available() else torch.device('cpu')


class NodeLevelGNN(pl.LightningModule):

    def __init__(self, model_name, dsParams : dmbioProtDataSetParams, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.dsParams = dsParams

        if model_name == "MLP":
            self.model = dmBioModel.MLPModel(**model_kwargs)
        else:
            self.model = dmBioModel.GNNModel(**model_kwargs)

        self.loss_module = nn.BCELoss() 
    def setup(self):
        self.dataset = dmbioProtDataSet(self.dsParams)
        self.traing_batch, self.validation_batch, self.test_batch = self.dataset.split_train_test_validation()

    def train_dataloader(self):
        return self.traing_batch
    
    def val_dataloader(self):
        return self.validation_batch
    
    def test_dataloader(self):
        return self.test_batch

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        pred = nn.Sigmoid()(x)
        loss = self.loss_module(pred, data.y)

        pred = pred.squeeze()
        pred = torch.tensor([0 if each<0.5 else 1 for each in pred], dtype=torch.int32)
        y_int = data.y.to(torch.int64)
        y_int = y_int.squeeze()
        t_p = ((pred == 1) & (y_int == 1)).sum()
        t_n = ((pred == 0) & (y_int == 0)).sum()
        actual_p = (y_int == 1).sum()
        actual_n = (y_int == 0).sum()
        f_p = ((pred == 1) & (y_int == 0)).sum()
        f_n = ((pred == 0) & (y_int == 1)).sum()
        tpr = t_p / (t_p + f_n)
        accuracy = (t_p + t_n ) / (t_p + f_p + t_n + f_n)
        precision = t_p / (t_p + f_p)
        f_1 = 2 * (precision * tpr) / (precision + tpr)

        return loss , tpr , accuracy, precision, f_1

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        lr_scheduler = 
        {
            'scheduker' : optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch),
            'name': 'cosine_annealing'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss , tpr , accuracy, precision, f_1 = self.forward(batch)
        self.log('train_loss', loss)
        self.log('train_tpr', tpr)
        self.log('train_accuracy', accuracy)
        self.log('train_precision', precision)
        self.log('train_f1', f_1)

        return loss

    def validation_step(self, batch, batch_idx):
        _ , tpr , accuracy, precision, f_1 = self.forward(batch)
        self.log('val_tpr', tpr)
        self.log('val_accuracy', accuracy)
        self.log('val_precision', precision)
        self.log('val_f1', f_1)

    def test_step(self, batch, batch_idx):
        _ , tpr , accuracy, precision, f_1 = self.forward(batch)
        self.log('test_tpr', tpr)
        self.log('test_accuracy', accuracy)
        self.log('test_precision', precision)
        self.log('test_f1', f_1)


