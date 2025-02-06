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
    def __init__(self, fileLogger,**model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = dmBioModel.GNNModel(**model_kwargs)
        self.loss_module = nn.BCELoss() 
        self.fileLogger = fileLogger

    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        x = self.model(x, edge_index)
        pred = nn.Sigmoid()(x)
        loss = self.loss_module(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.fileLogger.info(f"Training Loss: {loss}, Batch: {batch_idx}, Epoch: {self.current_epoch}")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, tpr = self._shared_eval_step(batch, batch_idx)
        metrices = {'val_loss': loss, 'val_tpr': tpr}
        self.log_dict(metrices, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.fileLogger.info(f"Validation Loss: {loss}, Validation TPR: {tpr}, Batch: {batch_idx}, Epoch: {self.current_epoch}")
        return metrices

    def test_step(self, batch, batch_idx):    
        loss, tpr = self._shared_eval_step(batch, batch_idx)
        metrices = {'test_loss': loss, 'test_tpr': tpr}
        self.log_dict(metrices, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrices

    def _shared_eval_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        # Forward pass
        x = self.model(x, edge_index)
        pred = torch.sigmoid(x)  
        
        # Compute loss
        loss = self.loss_module(pred, y)
        
        # Process predictions
        pred = pred.squeeze()
        pred = (pred >= 0.5).int().to(self.device)  # Directly create tensor on correct device

        # Convert y to the correct format
        y_int = y.to(torch.int64).squeeze()

        # Compute True Positives (TP) and False Negatives (FN)
        t_p = ((pred == 1) & (y_int == 1)).sum()
        f_n = ((pred == 0) & (y_int == 1)).sum()

        # Compute True Positive Rate (TPR)
        tpr = t_p / (t_p + f_n + 1e-8)  # Add small epsilon to avoid division by zero

        return loss, tpr


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        return [optimizer]

def main(fileLogger, hparams):
    L.seed_everything(42)
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f'cuda:{device_count - 1}') if torch.cuda.is_available() else torch.device('cpu')

    dataset_loader = dmbioProtDatasetloader(hparams.dsConfigPath)
    train, val , test = dataset_loader.split_train_test_validation(batch_size = int(hparams.batch))
    
    c_in = dataset_loader.num_features
    hidden_layers = [2048, 1024 , 512]
    root_dir = os.path.join(args.checkPtPath, f"ProtDNAAffinity_GNN")
    if not os.path.exists(hparams.checkPtPath):
        os.makedirs(root_dir, exist_ok=True)
    epochs = int(hparams.epoch)

    if torch.cuda.is_available():
        trainer =L.Trainer(default_root_dir=root_dir,
                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_tpr", filename='GNNModel')],
                                accelerator = "gpu" ,
                                max_epochs = epochs,
                                devices= device_count,
                                enable_progress_bar = True
                                )
    else:
        trainer =L.Trainer(default_root_dir=root_dir,
                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_tpr", filename='GNNModel')],
                                accelerator = "cpu" ,
                                max_epochs = epochs,
                                enable_progress_bar = True
                                )
    model = dmbioProtAffinityGNN(fileLogger, c_in =c_in, c_hiddens = hidden_layers)
    model.to(device)
    trainer.fit(model, train, val)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dsConfigPath", default=None)
    parser.add_argument("--checkPtPath", default=None)
    parser.add_argument("--batch", default=1)
    parser.add_argument("--epoch", default=10)
    args = parser.parse_args()

    logger = logging.getLogger("dmbioProtAffinityGNN_looger")
    logger.setLevel(logging.INFO)
    # Create a file handler
    if not os.path.exists(args.checkPtPath + "/logs"):
        os.makedirs(args.checkPtPath+ "/logs", exist_ok=True)

    log_dir = os.path.join(args.checkPtPath, f"logs/training_dmbioProtAffinityGNN_{datetime.now()}.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(file_handler)
    main(logger, args)