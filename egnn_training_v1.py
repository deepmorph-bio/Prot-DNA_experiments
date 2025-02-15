import lightning as L
import torch
import torch.nn as nn
import Src.model.egnn_clean as egnn
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from Src.data.dmBioProtDatasetLoader import dmbioProtDatasetloader
import traceback
from argparse import ArgumentParser
import os 
import logging
from datetime import datetime
import os


class dmbioProtAffinityEGNN(L.LightningModule):
    def __init__(self, fileLogger, lr_steps_per_epoch ,**model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = egnn.EGNN(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss()
        self.fileLogger = fileLogger
        self.lr_steps_per_epoch = lr_steps_per_epoch

    def training_step(self, batch, batch_idx):
        x, edge_index, pos, edge_attr ,y = batch.x, batch.edge_index, batch.pos, batch.edge_attr, batch.y
        h, x = self.model(x, pos, edge_index, edge_attr)
        loss = self.loss_module(h, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.fileLogger.info(f"Training Loss: {loss}, Batch: {batch_idx}, Epoch: {self.current_epoch}")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, tpr,f_n, t_p = self._shared_eval_step(batch, batch_idx)
        metrices = {'val_loss': loss, 'val_tpr': tpr, 'val_f_n': f_n, 'val_t_p': t_p}
        self.log_dict(metrices, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.fileLogger.info(f"Validation Loss: {loss}, Validation TPR: {tpr}, False Neg. {f_n} ,Batch: {batch_idx}, Epoch: {self.current_epoch}")
        return metrices

    def test_step(self, batch, batch_idx):    
        loss, tpr, f_n, t_p = self._shared_eval_step(batch, batch_idx)
        metrices = {'test_loss': loss, 'test_tpr': tpr, 'test_f_n': f_n, 'test_t_p': t_p}
        self.log_dict(metrices, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrices

    def _shared_eval_step(self, batch, batch_idx):
        x, edge_index, pos, edge_attr ,y = batch.x, batch.edge_index, batch.pos, batch.edge_attr, batch.y
        h, x = self.model(x, pos, edge_index, edge_attr)
        
        # Compute loss
        loss = self.loss_module(h, y)
        
        # Process predictions
        pred = (torch.sigmoid(h) >= 0.5).int().to(self.device) # Directly create tensor on correct device

        # Convert y to the correct format
        y_int = y.to(torch.int64).squeeze()

        # Compute True Positives (TP) and False Negatives (FN)
        t_p = ((pred == 1) & (y_int == 1)).sum()
        f_n = ((pred == 0) & (y_int == 1)).sum()

        # Compute True Positive Rate (TPR)
        tpr = t_p / (t_p + f_n + 1e-8)  # Add small epsilon to avoid division by zero

        return loss, tpr , f_n , t_p

    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
        #optimizer = torch.optim.Adam(self.parameters(), lr=4e-4, weight_decay=1e-4)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch = self.lr_steps_per_epoch , epochs=self.trainer.max_epochs),
            'interval': 'step'
        }
        return [optimizer] , [lr_scheduler]

def main(fileLogger, hparams):
    L.seed_everything(42)
    
    dataset_loader = dmbioProtDatasetloader(hparams.dsConfigPath)
    train, val , test = dataset_loader.split_train_test_validation(batch_size = int(hparams.batch))
    
    root_dir = os.path.join(hparams.checkPtPath, f"ProtDNAAffinity_EGNN")
    if not os.path.exists(hparams.checkPtPath):
        os.makedirs(root_dir, exist_ok=True)
    epochs = int(hparams.epoch)
    
    fileLogger.info(f"Training EGNN, train set length:{len(train)}")

    trainer =L.Trainer(default_root_dir=root_dir,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_tpr", filename='EGNNModel'),
                            EarlyStopping('val_loss', min_delta=0.000001)],
                            max_epochs = epochs,
                            enable_progress_bar = True
                        )

    if int(hparams.version)!= 99:
        model_path = os.path.join(hparams.checkPtPath,"ProtDNAAffinity_EGNN", "lightning_logs",f"version_{hparams.version}","checkpoints","EGNNModel.ckpt")
        if os.path.exists(model_path):
            logger.info(f"Loading Model from {model_path}")
            model = dmbioProtAffinityEGNN.load_from_checkpoint(model_path, fileLogger = fileLogger)
        else:
            model = dmbioProtAffinityEGNN(fileLogger, len(train) ,in_node_nf=dataset_loader.num_features, hidden_nf=512, out_node_nf=1, in_edge_nf=1)
    else:
        model = dmbioProtAffinityEGNN(fileLogger, len(train) ,in_node_nf=dataset_loader.num_features, hidden_nf=512, out_node_nf=1, in_edge_nf=1)

    trainer.fit(model, train, val)

def test(fileLogger,hparams):
    L.seed_everything(42)
    
    dataset_loader = dmbioProtDatasetloader(hparams.dsConfigPath)
    train, val , test = dataset_loader.split_train_test_validation(batch_size = int(hparams.batch))
    epochs = int(hparams.epoch)

    root_dir = os.path.join(hparams.checkPtPath, f"ProtDNAAffinity_EGNN")
    if not os.path.exists(hparams.checkPtPath):
        os.makedirs(root_dir, exist_ok=True)

    trainer =L.Trainer(default_root_dir=root_dir,
                        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_tpr", filename='EGNNModel'),
                        EarlyStopping('val_loss', min_delta=0.000001)],
                        max_epochs = epochs,
                        enable_progress_bar = True
                    )

    if int(hparams.version)!= 99:
        model_path = os.path.join(hparams.checkPtPath,"ProtDNAAffinity_EGNN", "lightning_logs",f"version_{hparams.version}","checkpoints","EGNNModel.ckpt")
        if os.path.exists(model_path):
            logger.info(f"Loading Model from {model_path}")
            model = dmbioProtAffinityEGNN.load_from_checkpoint(model_path, fileLogger = fileLogger)
            test_result = trainer.test(model, test)
            logger.info(f"Test Results:{test_result}")
        else:
            logger.error(f"Model not found {model_path}")
    else:
        logger.error("Invalid version or version not provided")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = ArgumentParser()
    parser.add_argument("--dsConfigPath", default=None)
    parser.add_argument("--checkPtPath", default=None)
    parser.add_argument("--batch", default=1)
    parser.add_argument("--epoch", default=10)
    parser.add_argument("--version", default=-99)
    parser.add_argument("--test", default=False)

    args = parser.parse_args()

    logger = logging.getLogger("dmbioProtAffinityEGNN_looger")
    logger.setLevel(logging.INFO)
    # Create a file handler
    if not os.path.exists(args.checkPtPath + "/logs"):
        os.makedirs(args.checkPtPath+ "/logs", exist_ok=True)

    log_dir = os.path.join(args.checkPtPath, f"logs/training_dmbioProtAffinityEGNN_{datetime.now()}.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(file_handler)

    try:
        if args.test:
            test(logger, args)
        else:
            main(logger, args)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
