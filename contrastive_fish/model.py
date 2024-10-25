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
from contrastive_fish.dataloader import *

class SimCLR(LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(num_classes=4 * hidden_dim)  # Output of last linear layer
        
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        # Metrics
        self.train_acc_top1 = Accuracy(top_k=1, task='multiclass', num_classes=7)
        self.train_acc_top5 = Accuracy(top_k=5, task='multiclass', num_classes=7)
        self.val_acc_top1 = Accuracy(top_k=1, task='multiclass', num_classes=7)
        self.val_acc_top5 = Accuracy(top_k=5, task='multiclass', num_classes=7)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr / 50)
        return [optimizer], [lr_scheduler]
    
    def info_nce_loss(self, batch, mode='train'):
        """
        Computes the InfoNCE loss for SimCLR.
        
        Args:
            batch (dict): A batch from the DataLoader containing 'contrastive_image'.
            mode (str): 'train' or 'val' for logging purposes.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        # Extract contrastive images
        contrastive_images = batch.get('contrastive_image', None)
        if contrastive_images is None:
            raise ValueError("Batch does not contain 'contrastive_image' key.")
        
        batch_size, n_views, C, H, W = contrastive_images.shape  # [batch_size, n_views, C, H, W]
        
        # Reshape to [batch_size * n_views, C, H, W]
        contrastive_images = contrastive_images.view(-1, C, H, W)
        
        # Pass through the convnet
        feats = self.convnet(contrastive_images)  # [batch_size * n_views, hidden_dim]
        
        # Normalize features
        feats = F.normalize(feats, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(feats, feats.T)  # [batch_size * n_views, batch_size * n_views]
        
        # Mask out self-similarity
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # Positive pairs: assume n_views=2, positives are batch_size apart
        # For example, if n_views=2, then for sample i, view1 is at i*2 and view2 is at i*2 +1
        # Thus, positives for view1 are view2 and vice versa
        if n_views != 2:
            raise ValueError("n_views must be 2 for this implementation.")
        
        # Create labels for contrastive loss
        labels = torch.cat([torch.arange(batch_size) + i * batch_size for i in range(n_views)], dim=0).to(feats.device)

        
        # Scale similarities by temperature
        logits = similarity_matrix / self.hparams.temperature
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Logging loss
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute accuracy metrics
        # For top-1 accuracy, the correct class is the positive pair
        preds = logits.argmax(dim=1)
        correct_top1 = (preds == labels).float().mean()
        self.log(f"{mode}_acc_top1", correct_top1, on_step=True, on_epoch=True, prog_bar=True)
        
        # For top-5 accuracy
        top5 = torch.topk(logits, k=5, dim=1).indices
        correct_top5 = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean()
        self.log(f"{mode}_acc_top5", correct_top5, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')


def train_simclr(batch_size, max_epochs=500, unlabeled_dataloader=None, labeled_dataloader=None, **kwargs):
    """
    Trains the SimCLR model using PyTorch Lightning.
    
    Args:
        batch_size (int): The batch size for training.
        max_epochs (int): The number of epochs to train.
        unlabeled_dataloader (DataLoader): DataLoader for unlabeled data.
        labeled_dataloader (DataLoader): DataLoader for labeled data (used as validation).
        **kwargs: Additional keyword arguments for SimCLR.
    
    Returns:
        SimCLR: The trained SimCLR model.
    """
    # Define checkpoint path
    CHECKPOINT_PATH = './checkpoints'  # Replace with your desired path
    device = torch.device('mps' if torch.backends.mps.is_available()  else 'cpu')
    
    # Initialize Trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
        accelerator="mps" if device.type == "cuda" else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
            LearningRateMonitor('epoch')
        ],
        logger=pl.loggers.TensorBoardLogger('lightning_logs', name='SimCLR')  # Optional: Add a logger
    )
    
    # Disable default hyperparameter logging
    trainer.logger._default_hp_metric = None
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        if unlabeled_dataloader is None or labeled_dataloader is None:
            raise ValueError("Both 'unlabeled_dataloader' and 'labeled_dataloader' must be provided.")
        
        # Set seed for reproducibility
        pl.seed_everything(42)
        
        # Initialize the model
        model = SimCLR(hidden_dim=kwargs.get('hidden_dim', 128),
                      lr=kwargs.get('lr', 5e-4),
                      temperature=kwargs.get('temperature', 0.07),
                      weight_decay=kwargs.get('weight_decay', 1e-4),
                      max_epochs=max_epochs)
        
        # Train the model
        trainer.fit(model, train_dataloaders=unlabeled_dataloader, val_dataloaders=labeled_dataloader)
        
        # Load the best checkpoint
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        if best_checkpoint:
            model = SimCLR.load_from_checkpoint(best_checkpoint)
        else:
            print("No checkpoint was saved.")
    
    return model

if __name__ == '__main__':
    
    labeled_dataloader, unlabeled_dataloader = get_datasets()
    
    # Define batch size and number of workers
    batch_size = 4
    NUM_WORKERS = 4  # Adjust based on your system's capabilities

    # Initialize and train the SimCLR model
    simclr_model = train_simclr(
            batch_size=batch_size,
            hidden_dim=128,
            lr=5e-4,
            temperature=0.07,
            weight_decay=1e-4,
            max_epochs=1,
            unlabeled_dataloader=unlabeled_dataloader,
            labeled_dataloader=labeled_dataloader
        )