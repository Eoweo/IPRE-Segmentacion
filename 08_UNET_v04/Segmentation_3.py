import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from model import UNet
from dataset import MainDataset, load_jpg_dataset_generator, initialize_patient_splits
import parameter as p


class UNetLightning(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, lr=3e-4):
        super(UNetLightning, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = UNet(in_channels, out_channels)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x, y.squeeze(1)
        print(f"Batch {batch_idx} running on device: {x.device}")  # Check device
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        dice_score = self.dice_coeff(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("train_dice", dice_score, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("train_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x, y.squeeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        dice_score = self.dice_coeff(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_dice", dice_score, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def dice_coeff(self, y_hat, y):
        y_hat = (y_hat > 0).float()
        intersection = (y_hat * y).sum()
        return (2. * intersection) / (y_hat.sum() + y.sum() + 1e-6)
    
    def accuracy(self, y_hat, y):
        y_hat = (y_hat > 0).float()
        correct = (y_hat == y).sum()
        return correct.float() / y.numel()

    def plot_predictions(self, loader, device='cuda'):
        self.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x
                y = y.squeeze(1)
                y_hat = self(x)
                y_hat = (y_hat > 0).float().squeeze(1)
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(x[0].cpu().numpy().squeeze(), cmap="gray")
                axes[0].set_title("Input Image")
                axes[0].axis("off")
                
                axes[1].imshow(y[0].cpu().numpy().squeeze(), cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                
                axes[2].imshow(y_hat[0].cpu().numpy().squeeze(), cmap="gray")
                axes[2].set_title("Prediction")
                axes[2].axis("off")
                
                wandb.log({"Model Predictions": wandb.Image(fig)})
                plt.close(fig)
                break


class LungSegmentationDataModule(L.LightningDataModule):
    def __init__(self, batch_size=p.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        patient_splits = initialize_patient_splits(p.PATH_CT_MARCOPOLO)
        train_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, patient_splits, dataset_type="test" , target_size = p.RESIZE_VALUE, block_id=p.BLOCK_ID)
        test_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, patient_splits, dataset_type="train", target_size = p.RESIZE_VALUE, block_id=p.BLOCK_ID)
        
        self.train_dataset = MainDataset(train_generator, augmentation=True, dataset_type="Training")
        self.val_dataset = MainDataset(test_generator, augmentation=False, dataset_type="Test")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=19)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=19)

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":

    wandb_logger = WandbLogger(project="unet-lightning-lung-segmentation", log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    
    data_module = LungSegmentationDataModule()
    model = UNetLightning(lr=p.LEARNING_RATE)
    
    #early_stop = L.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
    trainer = L.Trainer(logger=wandb_logger, 
                         callbacks=[checkpoint_callback], 
                         max_epochs=p.EPOCHS, 
                         accelerator="gpu",
                         devices=3,
                         log_every_n_steps=10
                         )
    trainer.fit(model, data_module)
        
    model.plot_predictions(data_module.val_dataloader())
    
    model_path = "model_final.ckpt"
    trainer.save_checkpoint(model_path)
    wandb.save(model_path)
    wandb.save("model_final.pth")
    wandb.finish()