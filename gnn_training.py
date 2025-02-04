import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import pytorch_lightning as pl
import Src.model.dmBioGNN as dmBioModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from Src.data.dmbioProtDataset import dmbioProtDataSetParams, dmbioProtDataSet
import traceback
from argparse import ArgumentParser
import configparser
import os 


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, epoch, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.c_in = model_kwargs['c_in']
        self.hidden_layers = model_kwargs['c_hiddens']
        self.c_out = model_kwargs['c_out']
        self.epoch = epoch

        if model_name == "MLP":
            self.model = dmBioModel.MLPModel(**model_kwargs)
        else:
            self.model = dmBioModel.GNNModel(**model_kwargs)

        self.loss_module = nn.BCELoss() 

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
        lr_scheduler = {
            'scheduler' : optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch),
            'name': 'cosine_annealing'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss , tpr , accuracy, precision, f_1 = self.forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_tpr', tpr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('train_f1', f_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _ , tpr , accuracy, precision, f_1 = self.forward(batch)
        # self.log('val_tpr', tpr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_f1', f_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        _ , tpr , accuracy, precision, f_1 = self.forward(batch)
        #self.log('test_tpr', tpr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('test_f1', f_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

def train(training_batch, validation_batch, test_batch, num_features ,args):
    pl.seed_everything(42)
    device_count = torch.cuda.device_count() - 1 if torch.cuda.is_available() else 0
    device = torch.device(f'cuda:{device_count}') if torch.cuda.is_available() else torch.device('cpu')

    c_in = num_features
    hidden_layers = [2048, 1024 , 512]
    root_dir = os.path.join(args.checkPtPath, f"ProtDNAAffinity_{ args.Model}")
    if not os.path.exists(args.checkPtPath):
        os.makedirs(root_dir, exist_ok=True)
    epochs = int(args.epoch) 
    if torch.cuda.is_available():
        trainer =pl.Trainer(default_root_dir=root_dir,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_precision")],
                            accelerator = "gpu",
                            max_epochs = epochs,
                            devices= device_count,
                            enable_progress_bar = True
                            )
    else:
        trainer = pl.Trainer(default_root_dir=root_dir,
                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_precision")],
                accelerator = "cpu",
                max_epochs = epochs,
                enable_progress_bar = True
                )

    
    pl.seed_everything()
    model = NodeLevelGNN(model_name = args.Model, epoch = epochs, c_in =c_in, c_hiddens = hidden_layers, c_out = 1)
    trainer.fit(model, training_batch, validation_batch)

def test(args):
    pretrained_filename = os.path.join(args.checkPtPath, f"ProtDNAAffinity_{ args.Model}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...WIP")
    else:
        print("Pre trained model not found, exiting..")
        return


if __name__ == "__main__":
    try:
        parser = ArgumentParser()
        parser.add_argument("--dsConfigPath", default=None)
        parser.add_argument("--checkPtPath", default=None)
        parser.add_argument("--Model", default=None)
        parser.add_argument("--train", default=True)
        parser.add_argument("--epoch", default=10)
        args = parser.parse_args()

        config = configparser.ConfigParser()
        config.read(args.dsConfigPath)
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
        dmdataset = dmbioProtDataSet(dsParams)
        print("Loading Dataset!")
        dmdataset.load_all()
        training_batch, validation_batch, test_batch = dmdataset.split_train_test_validation()
        print(f"Complete loading Dataset! Graph Count: {len(dmdataset.targets)}")
        num_features = dmdataset.dataset[0].num_features

        if args.train:
            train(training_batch, validation_batch, test_batch, num_features, args)
        else:
            test(args)
    except Exception as e:
        print(f'***Exception** : {e}')
        traceback.print_exc()