import lightning.pytorch as L
import io
import torch
import imageio

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
import os

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
        print("inicio de plot")
        images = []
        gif_frames = []
    
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).squeeze(1)  # Remove channel dimension
                y_hat = self(x)  # Use self instead of model
    
                x_np = x[:, 0].cpu().numpy()  
                y_np = y.cpu().numpy()  
                y_hat_np = y_hat.cpu().numpy()  # Ensure numpy conversion
    
                for i in range(len(x_np)):
                    images.append(wandb.Image(x_np[i], caption=f"Input - Sample {i}", mode="L"))
                    images.append(wandb.Image(y_np[i], caption=f"Ground Truth - Sample {i}", mode="L"))
                    images.append(wandb.Image(y_hat_np[i], caption=f"Prediction - Sample {i}", mode="L"))
    
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(x_np[i], cmap="gray")
                    axes[0].set_title("Input Image")
                    axes[0].axis("off")
    
                    axes[1].imshow(y_np[i], cmap="gray")
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis("off")
    
                    axes[2].imshow(y_hat_np[i], cmap="gray")
                    axes[2].set_title("Prediction")
                    axes[2].axis("off")
    
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    plt.close()
                    buf.seek(0)
                    gif_frames.append(imageio.imread(buf))
        
        self.logger.experiment.log({"Segmentation Results": images})

        gif_path = "predictions_over_epochs.gif"
        imageio.mimsave(gif_path, gif_frames, duration=0.5)
        self.logger.experiment.log({"Predictions GIF": wandb.Video(gif_path, fps=2, format="gif")})


    def get_model_name(base_name="v0", folder=".", ext=".ckpt"):
        """Find the next available zip file name with incremental numbering."""
        version = 1
        while True:
            name = f"model_{base_name}{version:02d}{ext}"
            if not os.path.exists(os.path.join(folder, name)):
                return name
            version += 1

class LungSegmentationDataModule(L.LightningDataModule):
    def __init__(self, batch_size=p.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        patient_splits = initialize_patient_splits(p.PATH_CT_MARCOPOLO)
        train_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, patient_splits, dataset_type="train", target_size=p.RESIZE_VALUE, block_id=p.BLOCK_ID)
        test_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, patient_splits, dataset_type="test", target_size=p.RESIZE_VALUE, block_id=p.BLOCK_ID)
        
        self.train_dataset = MainDataset(train_generator, augmentation=True, dataset_type="Training")
        self.val_dataset = MainDataset(test_generator, augmentation=False, dataset_type="Test")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=p.WORKERS)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=p.WORKERS)

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="unet-lightning-lung-segmentation", log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    
    data_module = LungSegmentationDataModule()
    
    if p.USE_PRETRAINED_MODEL:
        model = UNetLightning.load_from_checkpoint("model_final.ckpt")
        print("Loaded pre-trained model.")
        
        if not p.RE_TRAIN_MODEL:
            print("Testing pre-trained model performance.")
            trainer = L.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], accelerator="gpu", devices=3, log_every_n_steps=10)
            trainer.validate(model, data_module)
        else:
            print("Re-training the pre-trained model.")
            trainer = L.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], max_epochs=p.EPOCHS, accelerator="gpu", devices=2)#, log_every_n_steps=10)
            trainer.fit(model, data_module)
    else:
        model = UNetLightning(lr=p.LEARNING_RATE)
        trainer = L.Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], max_epochs=p.EPOCHS, accelerator="gpu", devices=2)#, log_every_n_steps=10)
        trainer.fit(model, data_module)
    
    if p.SAVE_PLOTS:
        model.plot_predictions(data_module.val_dataloader())

    if p.SAVE_MODEL:
        model_name = model.get_model_name(folder=p.RESULT_DIR)
        model_path = os.path.join(p.RESULT_DIR, "model.ckpt")
        trainer.save_checkpoint(model_path)
        wandb.save(model_path)
        torch.save(model.state_dict(), "model_final.pth")
        wandb.save("model_final.pth")
        os.remove("model_final.pth")
        print("Model saved and uploaded.")
    
    wandb.finish()
