import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from contrastive_learning.dataloader import *

import wandb
from pytorch_lightning.loggers import WandbLogger

# Initialize WandbLogger
WANDB_LOGGER = WandbLogger(
    project='SimCLR_Project',     
    name='SimCLR_Run',           
    log_model='all',               # Log all models
    save_dir='wandb_logs'
)

class SimCLR(pl.LightningModule):
    """
    SimCLR model implemented using PyTorch Lightning.

    Attributes:
        convnet (nn.Module): Backbone convolutional network with projection head.
        hidden_dim (int): Dimension of the projection head's output.
        lr (float): Learning rate for the optimizer.
        temperature (float): Temperature parameter for InfoNCE loss.
        weight_decay (float): Weight decay for the optimizer.
        max_epochs (int): Maximum number of training epochs.
    """
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        """
        Initializes the SimCLR model.

        Args:
            hidden_dim (int): Dimension of the projection head's output.
            lr (float): Learning rate for the optimizer.
            temperature (float): Temperature parameter for InfoNCE loss.
            weight_decay (float): Weight decay for the optimizer.
            max_epochs (int, optional): Maximum number of training epochs.
        """
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            list: Optimizer(s).
            list: Scheduler(s).
        """
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        """
        Computes the InfoNCE loss for a batch of images.

        Args:
            batch (tuple): Tuple containing images and labels.
            mode (str, optional): Mode of operation ('train' or 'val').

        Returns:
            Tensor: Computed InfoNCE loss.
        """
        contrastive_images, _ = batch  # contrastive_images shape: (batch_size, n_views, C, H, W)
        batch_size, n_views, C, H, W = contrastive_images.shape
        # Reshape to (batch_size * n_views, C, H, W)
        imgs = contrastive_images.view(batch_size * n_views, C, H, W)

        # Encode all images
        feats = self.convnet(imgs)  # feats shape: (batch_size * n_views, hidden_dim)

        # Normalize features
        feats = F.normalize(feats, dim=1)

        # Compute cosine similarity matrix
        cos_sim = torch.matmul(feats, feats.T)  # shape: (2*batch_size, 2*batch_size)

        # Mask to exclude self-similarity
        mask = torch.eye(cos_sim.size(0), dtype=torch.bool, device=self.device)
        cos_sim.masked_fill_(mask, -9e15)

        # Compute positive mask
        # Assuming first n_views are from one augmentation and the next n_views from another
        # Adjust based on your data batching strategy
        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        pos_mask = torch.cat([pos_mask, pos_mask], dim=0)
        pos_mask = pos_mask.repeat(1, n_views).view(-1, pos_mask.size(1)*n_views)

        # Extract positive similarities
        pos_sim = cos_sim[pos_mask].view(batch_size * n_views, -1)

        # Compute loss
        loss = -torch.log(torch.exp(pos_sim / self.hparams.temperature) / torch.sum(torch.exp(cos_sim / self.hparams.temperature), dim=1))
        loss = loss.mean()

        # Logging loss
        self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Get ranking position of positive example
        # Not implemented here as it's more involved; can be added based on specific needs

        return loss

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch (tuple): Tuple containing images and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss.
        """
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch (tuple): Tuple containing images and labels.
            batch_idx (int): Index of the batch.
        """
        self.info_nce_loss(batch, mode='val')


def train_simclr(batch_size, max_epochs=500, unlabeled_dataloader=None, labeled_dataloader=None, **kwargs):
    """
    Trains the SimCLR model using PyTorch Lightning.

    Args:
        batch_size (int): Number of samples per batch.
        max_epochs (int, optional): Maximum number of training epochs.
        unlabeled_dataloader (DataLoader): DataLoader for unlabeled data.
        labeled_dataloader (DataLoader): DataLoader for labeled data.
        **kwargs: Additional hyperparameters (hidden_dim, lr, temperature, weight_decay).

    Returns:
        SimCLR: Trained SimCLR model.
    """
    
    CHECKPOINT_PATH = './checkpoints'  # Replace with your desired path
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Initialize WandbLogger
    WANDB_LOGGER = WandbLogger(
        project='SimCLR_Project',     
        name='SimCLR_Run',           
        log_model='all',               # Log all models
        save_dir='wandb_logs'
    )
    
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode='min', monitor='train_loss'),
            LearningRateMonitor('epoch')
        ],
        logger=[pl.loggers.TensorBoardLogger('lightning_logs', name='SimCLR'), WANDB_LOGGER],
        log_every_n_steps=10,  # Adjust as needed
    )
    
    trainer.logger._default_hp_metric = False  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42)  # To be reproducible
        model = SimCLR(
            hidden_dim=kwargs.get('hidden_dim', 128),
            lr=kwargs.get('lr', 5e-4),
            temperature=kwargs.get('temperature', 0.07),
            weight_decay=kwargs.get('weight_decay', 1e-4),
            max_epochs=max_epochs
        )
                            
        # Train the model
        trainer.fit(model, train_dataloaders=unlabeled_dataloader, val_dataloaders=labeled_dataloader)
        
        # Load the best checkpoint
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        if best_checkpoint:
            model = SimCLR.load_from_checkpoint(best_checkpoint)
            print(f"Loaded best model from {best_checkpoint}")
        else:
            print("No checkpoint was saved.")

    return model


if __name__ == '__main__':
    # Define batch size and number of workers
    batch_size = 16
    NUM_WORKERS = 4  # Adjust based on your system's capabilities

    # Get DataLoaders
    labeled_dataloader, unlabeled_dataloader = get_datasets(batch_size=batch_size)

    # Initialize and train the SimCLR model
    simclr_model = train_simclr(
        batch_size=batch_size,
        hidden_dim=128,
        lr=5e-4,
        temperature=0.07,
        weight_decay=1e-4,
        max_epochs=30,
        unlabeled_dataloader=unlabeled_dataloader,
        labeled_dataloader=labeled_dataloader,
    )