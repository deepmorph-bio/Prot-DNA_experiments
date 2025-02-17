import lightning as L
from watermark import watermark
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
import time
from torch import tensor
from torchmetrics.classification import BinaryRecall
from tqdm import tqdm

def training_loop(model, criterion, optimizer, scheduler ,train_loader, val_loader, epochs, fabric, device, fileLogger):
    loss_history = [0.0] * epochs
    train_tpr_history = [0.0] * epochs
    val_loss_history = [0.0] * epochs
    val_tpr_history = [0.0] * epochs

    metric = BinaryRecall()
    metric.to(device)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        #for batch in train_loader:
        for batch_idx, batch in loop:
            x, edge_index, pos, edge_attr ,y = batch.x, batch.edge_index, batch.pos, batch.edge_attr, batch.y
            h, x = model(x, pos, edge_index, edge_attr)
            loss = criterion(h, y)
            optimizer.zero_grad()
            fabric.backward(loss)

            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                 # Process predictions
                pred = (torch.sigmoid(h).squeeze() >= 0.5).int().to(device) # Directly create tensor on correct device
                # Convert y to the correct format
                y_int = y.to(torch.int64).squeeze()
                loss_history[epoch] += loss.item()
                tpr  = metric(pred, y_int)
                train_tpr_history[epoch] += tpr.item()
                # Update progress bar
                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(loss=loss.item())

         ### MORE LOGGING
        model.eval()
        with torch.no_grad():
            loss_history[epoch] /= len(train_loader.dataset)
            train_tpr_history[epoch] /= len(train_loader.dataset)
            loop_val = tqdm(enumerate(train_loader), total=len(val_loader), leave=True)
            for batch_idx, batch in loop_val:
                x, edge_index, pos, edge_attr ,y = batch.x, batch.edge_index, batch.pos, batch.edge_attr, batch.y
                h, x = model(x, pos, edge_index, edge_attr)
                loss = criterion(h, y)
                pred = (torch.sigmoid(h).squeeze() >= 0.5).int().to(device) 
                y_int = y.to(torch.int64).squeeze()

                val_loss_history[epoch] += loss.item()
                tpr  = metric(pred, y_int)
                val_tpr_history[epoch] += tpr.item()
                loop_val.set_description(f"Validation:[{epoch}/{epochs}]")
                loop_val.set_postfix(loss=loss.item())

            val_loss_history[epoch] /= len(train_loader.dataset)
            val_tpr_history[epoch] /= len(train_loader.dataset) 

        fileLogger.info(f"Epoch {epoch}: Loss = {loss_history[epoch]:.4f}, \
        Recall = {train_tpr_history[epoch]:.4f} \
        Val Loss = {val_loss_history[epoch]:.4f}, \
        Recall = {val_tpr_history[epoch]:.4f}")        

    return loss_history, train_tpr_history, val_loss_history, val_tpr_history

def test_loop(model, test_loader, device, fileLogger):
    metric = BinaryRecall()
    metric.to(device)
    test_tpr_history = [0.0] * len(test_loader)
    with torch.no_grad():
        model.eval()
        i=0
        loop_val = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        for batch_idx, batch in loop_val:
            x, edge_index, pos, edge_attr ,y = batch.x, batch.edge_index, batch.pos, batch.edge_attr, batch.y
            h, x = model(x, pos, edge_index, edge_attr)
            pred = (torch.sigmoid(h).squeeze() >= 0.5).int().to(device) 
            y_int = y.to(torch.int64).squeeze()
            tpr  = metric(pred, y_int)
            test_tpr_history[i] += tpr.item()
            i+=1
            loop_val.set_description(f"Testing:")
            loop_val.set_postfix(TPR=tpr.item())

    fileLogger.info(f"Test Recall: {sum(test_tpr_history)/len(test_loader):.4f}")

def main(fileLogger, hparams):
    fileLogger.info(f"hparams: {hparams}")
    fileLogger.info(watermark(packages="torch,lightning", python=True))
    fileLogger.info(f"Torch CUDA available? {torch.cuda.is_available()}")

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f'cuda:{device_count - 1}') if torch.cuda.is_available() else torch.device('cpu')

    fabric = L.Fabric(accelerator="cuda", devices=device_count, precision="16-mixed")
    fabric.launch()

    torch.set_float32_matmul_precision('high')

    L.seed_everything(123)
    ##########################
    ### 1 Loading the Dataset
    ##########################
    fileLogger.info("Loading the dataset")
    dataset_loader = dmbioProtDatasetloader(hparams.dsConfigPath)
    train, val , test = dataset_loader.split_train_test_validation()

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(
        train, val, test)

    fileLogger.info(f"Train, Val, Test dataset sizes: {len(train)}, {len(val)}, {len(test)}" )
    
    hidden_nf = int(hparams.hidden_nf)
    n_layers = int(hparams.n_layers)
    
    
    #########################################
    ### 2 Initializing the Model
    #########################################

    model = egnn.EGNN(in_node_nf=dataset_loader.num_features,
     hidden_nf=hidden_nf, out_node_nf=1, 
     in_edge_nf=1,attention=True, 
     n_layers=n_layers)

    loss_module = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(
     model.parameters(), 
     lr=5e-4,
     momentum=0.9,
     weight_decay=1e-4)

    lr_steps_per_epoch = len(train)
    max_epochs = int(hparams.epoch)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
            max_lr=1e-3, 
            steps_per_epoch = lr_steps_per_epoch , 
            epochs=max_epochs)

    model, optimizer = fabric.setup(model, optimizer)

    #########################################
    ### 3 Finetuning
    #########################################

    start = time.time()
    loss_history, tpr_history, val_loss_history, val_tpr_history = training_loop(model, 
    loss_module, 
    optimizer, 
    lr_scheduler, 
    train_loader,
    val_loader,
    max_epochs,
    fabric,
    device,
    fileLogger)

    end = time.time()
    elapsed = end-start
    fileLogger.info(f"Time elapsed {elapsed/60:.2f} min")
    fileLogger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    fileLogger.info(f"Memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    test_loop(model, test_loader, device, fileLogger)

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = ArgumentParser()
    parser.add_argument("--dsConfigPath", default=None)
    parser.add_argument("--checkPtPath", default=None)
    parser.add_argument("--epoch", default=10)
    parser.add_argument("--hidden_nf", default=768)
    parser.add_argument("--n_layers", default=10)

    args = parser.parse_args()

    logger = logging.getLogger("dmbioProtAffinityEGNN_looger")
    logger.setLevel(logging.INFO)
    # Create a file handler
    if not os.path.exists(args.checkPtPath + "/logs"):
        os.makedirs(args.checkPtPath+ "/logs", exist_ok=True)

    log_dir = os.path.join(args.checkPtPath, f"logs/fabric_training_dmbioProtAffinityEGNN_{datetime.now()}.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(file_handler)

    try:
        main(logger, args)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
