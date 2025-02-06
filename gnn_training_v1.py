import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import Src.model.dmBioGNN as dmBioModel
from lightning.pytorch.callbacks import ModelCheckpoint
from Src.data.dmBioProtDatasetLoader import dmbioProtDatasetloader
import traceback
from argparse import ArgumentParser
import os 
import logging
from datetime import datetime

class dmbioProtAffinityGNN(L.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = dmBioModel.GNNModel(**model_kwargs)
        self.loss_module = nn.BCELoss() 

    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        x = self.model(x, edge_index)
        pred = nn.Sigmoid()(x)
        loss = self.loss_module(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, tpr = self._shared_eval_step(batch, batch_idx)
        metrices = {'val_loss': loss, 'val_tpr': tpr}
        self.log_dict(metrices, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrices

    def test_step(self, batch, batch_idx):    
        loss, tpr = self._shared_eval_step(batch, batch_idx)
        metrices = {'test_loss': loss, 'test_tpr': tpr}
        self.log_dict(metrices, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrices

    def _shared_eval_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        x = self.model(x, edge_index)
        pred = nn.Sigmoid()(x)
        loss = self.loss_module(pred, y)
        pred = pred.squeeze()
        pred = torch.tensor([0 if each<0.5 else 1 for each in pred], dtype=torch.int32)
        y_int = y.to(torch.int64)
        y_int = y_int.squeeze()
        t_p = ((pred == 1) & (y_int == 1)).sum()
        f_n = ((pred == 0) & (y_int == 1)).sum()
        tpr = t_p / (t_p + f_n)
        return loss, tpr

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        return [optimizer]

def main(hparams):
    L.seed_everything(42)
    dataset_loader = dmbioProtDatasetloader(hparams.dsConfigPath)
    train, val , test = dataset_loader.split_train_test_validation(batch_size = int(hparams.batch))

    c_in = dataset_loader.num_features
    hidden_layers = [2048, 1024 , 512]
    root_dir = os.path.join(args.checkPtPath, f"ProtDNAAffinity_GNN")
    if not os.path.exists(hparams.checkPtPath):
        os.makedirs(root_dir, exist_ok=True)
    epochs = int(hparams.epoch)

    trainer =L.Trainer(default_root_dir=root_dir,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_tpr", filename='GNN-{epoch}-{val_tpr:.2f}')],
                            accelerator = "gpu" if torch.cuda.is_available() else "cpu",
                            max_epochs = epochs,
                            enable_progress_bar = True
                            )

    model = dmbioProtAffinityGNN(c_in =c_in, c_hiddens = hidden_layers)
    trainer.fit(model, train, val)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dsConfigPath", default=None)
    parser.add_argument("--checkPtPath", default=None)
    parser.add_argument("--batch", default=1)
    parser.add_argument("--epoch", default=10)
    args = parser.parse_args()
    main(args)