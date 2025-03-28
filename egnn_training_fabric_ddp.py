import lightning as L
from lightning.fabric.strategies.ddp import DDPStrategy
from watermark import watermark
import torch
import torch.nn as nn
import Src.model.egnn_clean as egnn
from Src.data.dmBioProtDatasetLoader import dmbioProtDatasetloader
import traceback
from argparse import ArgumentParser
import os 
import logging
from datetime import datetime
import time
from torch import tensor
from torchmetrics.classification import BinaryRecall, BinaryPrecision
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def save_checkpoint_callback(epoch, loss, model, optimizer, lr_scheduler, checkPtPath, version ,save_frequency=10):
    if (epoch + 1) % int(save_frequency) == 0:
        model_path = os.path.join(checkPtPath, "fabric", f"version_{version}", "checkpoints")
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        model_path = os.path.join(model_path, f"EGNNModel_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss,
            'hidden_nf': model.hidden_nf,
            'n_layers': model.n_layers
        }, model_path)

def plot_training_history(loss_history, tpr_history, val_loss_history, val_tpr_history, checkPtPath, version):
    # Example training history (replace with actual values)
    epochs = list(range(1, len(loss_history) + 1))

    # Create a DataFrame for easier plotting with seaborn
    history_df = pd.DataFrame({
        'Epoch': epochs * 2,  # Duplicate epochs for train & val
        'Loss': loss_history + val_loss_history,
        'TPR': tpr_history + val_tpr_history,
        'Type': ['Train'] * len(loss_history) + ['Validation'] * len(val_loss_history)
    })

    # Set Seaborn style
    sns.set_style("whitegrid")

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Loss
    sns.lineplot(data=history_df, x='Epoch', y='Loss', hue='Type', ax=ax1, marker='o', palette=['blue', 'orange'])
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create second y-axis for TPR
    ax2 = ax1.twinx()
    sns.lineplot(data=history_df, x='Epoch', y='TPR', hue='Type', ax=ax2, marker='s', linestyle='dashed', palette=['green', 'red'])
    ax2.set_ylabel('TPR', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Titles and layout
    plt.title("Training History: Loss & TPR")
    fig.tight_layout()

    image_path = os.path.join(checkPtPath, "fabric", f"version_{version}", "checkpoints","image")
    if not os.path.exists(image_path):
        os.makedirs(image_path, exist_ok=True)

    image_path = os.path.join(image_path, "training_history.png")
    plt.savefig(image_path)

def training_loop(model, criterion, optimizer, scheduler ,train_loader, val_loader, epochs, fabric, device, fileLogger, save_checkpoint_callback, **kwargs):
    loss_history = [0.0] * epochs
    train_tpr_history = [0.0] * epochs
    train_prec_history = [0.0] * epochs

    val_loss_history = [0.0] * epochs
    val_tpr_history = [0.0] * epochs
    val_prec_history = [0.0] * epochs

    metric = BinaryRecall()
    metric.to(device)

    metric_prec = BinaryPrecision()
    metric_prec.to(device)

    best_val_loss = float("inf")
    early_stopping_counter = 0
    
    checkptpath = kwargs.get("checkPtPath", "")
    version = kwargs.get("version", "1")
    save_frequency = kwargs.get("save_frequency", 10)
    patience = int(kwargs.get("patience", 0))
    min_delta = kwargs.get("min_delta", 0.00001)
    max_grad_norm = kwargs.get("max_grad_norm", 1.0)

    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        epoch_train_tprs = []
        epoch_train_prec = []
        # Only create progress bar on rank 0 process to avoid duplicate outputs
        is_main_process = fabric.global_rank == 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, disable=not is_main_process)
        #for batch in train_loader:
        model.train()
        for batch_idx, batch in loop:
            model.train()
            x, edge_index, pos, edge_attr ,y = batch.x.to(device), batch.edge_index.to(device), batch.pos.to(device), batch.edge_attr.to(device), batch.y.to(device)
            # Forward pass
            h, x = model(x, pos, edge_index, edge_attr)
            loss = criterion(h, y)
            optimizer.zero_grad()
            fabric.backward(loss)

            # Add gradient clipping before optimizer step - NEW
            # if max_grad_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            # Synchronize loss across processes for logging
            train_loss_item = torch.tensor(loss.item(), device=device)
            avg_train_loss = fabric.all_reduce(train_loss_item, reduce_op="mean")
            epoch_train_losses.append(avg_train_loss.item())

            model.eval()
            with torch.no_grad():
                # Process predictions
                pred = (torch.sigmoid(h).squeeze() >= 0.5).int().to(device) # Directly create tensor on correct device
                # Convert y to the correct format
                y_int = y.to(torch.int64).squeeze()
                tpr  = metric(pred, y_int)
                train_tpr_item = torch.tensor(tpr.item(), device=device)
                agg_tpr = fabric.all_reduce(train_tpr_item, reduce_op="mean")
                epoch_train_tprs.append(agg_tpr.item())

                prec  = metric_prec(pred, y_int)
                train_prec_item = torch.tensor(prec.item(), device=device)
                agg_prec = fabric.all_reduce(train_prec_item, reduce_op="mean")
                epoch_train_prec.append(agg_prec.item())

            # Update progress bar on main process only
            if is_main_process:
                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=avg_train_loss.item(), lr=scheduler.get_last_lr()[0])
                if batch_idx % 25 == 0:
                    fileLogger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {avg_train_loss.item():.4f}, Recall = {agg_tpr.item():.4f}, Precision = {agg_prec.item():.4f}")
     

        if is_main_process:
            loss_history[epoch] = sum(epoch_train_losses) / len(epoch_train_losses)
            train_tpr_history[epoch] = sum(epoch_train_tprs) / len(epoch_train_tprs)
            train_prec_history[epoch] = sum(epoch_train_prec) / len(epoch_train_prec)

        # Validation phases
        model.eval()
        val_losses = []
        val_tprs = []
        val_precs = []
        with torch.no_grad():
            loop_val = tqdm(enumerate(train_loader), total=len(val_loader), leave=True, disable=not is_main_process)
            for batch_idx, batch in loop_val:
                x, edge_index, pos, edge_attr ,y = batch.x, batch.edge_index, batch.pos, batch.edge_attr, batch.y
                h, x = model(x, pos, edge_index, edge_attr)
                val_loss = criterion(h, y)
                # Synchronize validation loss across processes
                val_loss_item = torch.tensor(val_loss.item(), device=device)
                avg_val_loss = fabric.all_reduce(val_loss_item, reduce_op="mean")
                val_losses.append(avg_val_loss.item())

                pred = (torch.sigmoid(h).squeeze() >= 0.5).int().to(device) 
                y_int = y.to(torch.int64).squeeze()

                val_tpr  = metric(pred, y_int)
                val_tpr_item = torch.tensor(val_tpr.item(), device=device)
                val_agg_tpr = fabric.all_reduce(val_tpr_item, reduce_op="mean")
                val_tprs.append(val_agg_tpr.item())

                val_prec  = metric_prec(pred, y_int)
                val_prec_item = torch.tensor(val_prec.item(), device=device)
                val_agg_prec = fabric.all_reduce(val_prec_item, reduce_op="mean")
                val_precs.append(val_agg_prec.item())


                if is_main_process:
                    loop_val.set_description(f"Validation:[{epoch+1}/{epochs}]")
                    loop_val.set_postfix(loss=avg_val_loss.item())
                    # Calculate validation metrics
            if val_losses:
                val_loss_history[epoch] = sum(val_losses) / len(val_losses)
                val_tpr_history[epoch] = sum(val_tprs) / len(val_tprs)
                val_prec_history[epoch] = sum(val_precs) / len(val_precs)



        if is_main_process:
            fileLogger.info(f"Epoch {epoch}: Loss = {loss_history[epoch]:.4f}, \
            Recall = {train_tpr_history[epoch]:.4f} \
            Val Loss = {val_loss_history[epoch]:.4f}, \
            Recall = {val_tpr_history[epoch]:.4f}, \
            Precision = {val_prec_history[epoch]:.4f}")        

            # Checkpoint Saving
            save_checkpoint_callback(epoch, val_loss_history[epoch], model, optimizer, scheduler, checkptpath, version, save_frequency)
            # Early Stopping
            if val_loss_history[epoch] < best_val_loss - min_delta:
                best_val_loss = val_loss_history[epoch]
                early_stopping_counter = 0  # Reset patience counter
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                fileLogger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
    return loss_history, train_tpr_history, val_loss_history, val_tpr_history

def test_loop(fabric, model, test_loader, device, fileLogger):
    is_main_process = fabric.global_rank == 0
    metric = BinaryRecall()
    metric.to(device)
    metric_prec = BinaryPrecision()
    metric_prec.to(device)

    test_tpr_history = [0.0] * len(test_loader)
    test_prec_history = [0.0] * len(test_loader)

    with torch.no_grad():
        model.eval()
        loop_val = tqdm(enumerate(test_loader), total=len(test_loader), leave=True, disable=not is_main_process)
        for batch_idx, batch in loop_val:
            x, edge_index, pos, edge_attr ,y = batch.x, batch.edge_index, batch.pos, batch.edge_attr, batch.y
            h, x = model(x, pos, edge_index, edge_attr)
            pred = (torch.sigmoid(h).squeeze() >= 0.5).int().to(device) 
            y_int = y.to(torch.int64).squeeze()

            tpr  = metric(pred, y_int)
            test_tpr_item = torch.tensor(tpr.item(), device=device)
            agg_tpr = fabric.all_reduce(test_tpr_item, reduce_op="mean")
            test_tpr_history[batch_idx] = agg_tpr.item()

            prec  = metric_prec(pred, y_int)
            test_prec_item = torch.tensor(prec.item(), device=device)
            agg_prec = fabric.all_reduce(test_prec_item, reduce_op="mean")
            test_prec_history[batch_idx] = agg_prec.item()


            if is_main_process:
                loop_val.set_description(f"Testing:")
                loop_val.set_postfix(TPR=tpr.item())
    
    if is_main_process:
        fileLogger.info(f"Test Recall: {sum(test_tpr_history)/len(test_loader):.4f}, Test Precision: {sum(test_prec_history)/len(test_loader):.4f}")

def main(fileLogger, hparams):
    fileLogger.info(f"hparams: {hparams}")
    fileLogger.info(watermark(packages="torch,lightning", python=True))
    fileLogger.info(f"Torch CUDA available? {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        logger.error("CUDA not available, Fabric training not possible, exitting..")
        return
        
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    #device = torch.device(f'cuda:{device_count - 1}') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device count: {device_count}, Device: {device}")

    if device_count > 1:
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
        logger.info(f'Device Count{device_count},  setting TORCH_DISTRIBUTED_DEBUG in INFO mode.')

    precision="16-mixed"

    if torch.cuda.is_bf16_supported() and hparams.bf16:
        logger.info("BF16 supported, setting precision to bf16 mixed")
        precision = "bf16-mixed"
    else:
        logger.info("BF16 either not supported or not requested, setting precision to 16 mixed")

    strategy = 'ddp_find_unused_parameters_true' if device_count > 1 else 'auto'
    fabric = L.Fabric(accelerator="cuda", devices=device_count, precision = precision, strategy = strategy)
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
    checkpoint = None
    if hparams.load_from_path and os.path.exists(hparams.load_from_path):
        logger.info(f"Loading checkpt from {hparams.load_from_path}")
        checkpoint = torch.load(hparams.load_from_path, map_location=device)
        if 'hidden_nf' in checkpoint:
            hidden_nf = checkpoint['hidden_nf']
        if 'n_layers' in checkpoint:
            n_layers = checkpoint['n_layers']


    model = egnn.EGNN(in_node_nf=dataset_loader.num_features,
     hidden_nf=hidden_nf, out_node_nf=1, 
     in_edge_nf=1,attention=True, 
     n_layers=n_layers)
    model.to(device)
    model.train()

    loss_module = nn.BCEWithLogitsLoss()

    
        

    # optimizer = torch.optim.SGD(
    #  model.parameters(), 
    #  lr=5e-4,
    #  momentum=0.9,
    #  weight_decay=1e-4)

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-16)

    learning_rate = float(hparams.lr)

    if hparams.optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=float(hparams.lr))
    elif hparams.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
        weight_decay=1e-5)

    
    # lr_steps_per_epoch = len(train)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
    #         max_lr=1e-3, 
    #         steps_per_epoch = lr_steps_per_epoch , 
    #         epochs=max_epochs)

    max_epochs = int(hparams.epoch)
    num_steps = max_epochs * len(train_loader)
    if hparams.lr_scheduler == "cosine":
        eta_min = float(hparams.eta_min)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min = eta_min)
    elif hparams.lr_scheduler == "cosine_warmrestart":
        eta_min = float(hparams.eta_min)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=50,           # First restart occurs after 50 epochs
                    T_mult=2,         # Each restart cycle is twice as long as the previous
                    eta_min=eta_min     # Minimum learning rate
                )
    elif hparams.lr_scheduler == "cyclic":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=float(hparams.lr),            # Your stable base learning rate
                    max_lr=0.04,               # Maximum learning rate (try ~2-3x your base rate)
                    step_size_up=5 * len(train_loader),  # Steps to reach max_lr (5 epochs)
                    step_size_down=15 * len(train_loader),  # Steps to return to base_lr (15 epochs)
                    mode='triangular2',        # Decrease amplitude by half each cycle
                    cycle_momentum=False       # Don't cycle momentum
                )
    else:
    # Use OneCycleLR for better convergence
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(hparams.lr),
            total_steps=num_steps,
            pct_start=0.3,
            div_factor=25,             # initial_lr = max_lr/div_factor
            final_div_factor=10000,    # final_lr = initial_lr/final_div_factor
            anneal_strategy='cos'
        )


    model, optimizer = fabric.setup(model, optimizer)

    if checkpoint:
        logger.info(f"Loading state dict from checkpoint")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()
        fileLogger.info(summary(model))

        load_optim_dict = bool(hparams.load_optim_dict)
        if load_optim_dict:
            logger.info(f"Loading optimizer state dict from checkpoint")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Instantiate callbacks
    save_checkpoint = lambda epoch, loss, model, optimizer, lr_scheduler, checkptpath, version, save_frequency: save_checkpoint_callback(epoch, loss, model, optimizer, lr_scheduler, checkptpath, version ,save_frequency)

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
    fileLogger,
    save_checkpoint,
    checkptpath = hparams.checkPtPath,
    version = hparams.version,
    save_frequency = hparams.save_frequency,
    min_delta  = hparams.es_min_delta,
    patience = hparams.es_patience,
    max_grad_norm = 1.0
    )

    end = time.time()
    elapsed = end-start
    fileLogger.info(f"Time elapsed {elapsed/60:.2f} min")
    fileLogger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    fileLogger.info(f"Memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    is_main_process = fabric.global_rank == 0
    test_loop(fabric, model, test_loader, device, fileLogger)
    
    if is_main_process:
        

        logger.info('******************************')
        logger.info(f'Loss history: {loss_history}')
        logger.info(f'TPR history: {tpr_history}')
        logger.info(f'Val Loss history: {val_loss_history}')
        logger.info(f'Val TPR history: {val_tpr_history}')
        logger.info('******************************')

        plot_training_history(loss_history, tpr_history, val_loss_history, val_tpr_history, hparams.checkPtPath, hparams.version)

if __name__ == "__main__":
    now = datetime.now()
    formatted_datetime = now.strftime("%d%m%H%M")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = ArgumentParser()
    parser.add_argument("--dsConfigPath", default=None)
    parser.add_argument("--checkPtPath", default=None)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--epoch", default=10)
    parser.add_argument("--hidden_nf", default=768)
    parser.add_argument("--n_layers", default=10)
    parser.add_argument("--version", default=formatted_datetime)
    parser.add_argument("--save_frequency", default=10)
    parser.add_argument("--es_min_delta", default=0.000001)
    parser.add_argument("--es_patience", default=3)
    parser.add_argument("--load_from_path", default=None)
    parser.add_argument("--load_optim_dict", default=False)
    parser.add_argument("--optim", default='SGD')
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--bf16", default=False)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--eta_min", default=0.001)
    args = parser.parse_args()

    logger = logging.getLogger("dmbioProtAffinityEGNN_looger")
    logger.setLevel(logging.INFO)
    # Create a file handler
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir, exist_ok=True)

    log_filepath = os.path.join(args.logdir, f"fabric_training_dmbioProtAffinityEGNN_{formatted_datetime}.log")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        logger.info(f"Run version: {args.version}")
        main(logger, args)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
