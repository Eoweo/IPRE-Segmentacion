import lightning.pytorch as L
import io
import torch
import numpy as np
import seaborn as sns
import imageio.v2 as imageio
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
import wandb
from model import UNet
from dataset import MainDataset, load_jpg_dataset_generator, initialize_patient_splits
import parameter as p
import os

class UNetLightning(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, lr=p.LEARNING_RATE):
        super(UNetLightning, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = UNet(in_channels, out_channels)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, id = batch
        x, y = x, y.squeeze(1)
        y = (y > 0.5).float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        dice_score = self.dice_coeff(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_dice", dice_score, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, id = batch
        x, y = x, y.squeeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        dice_score = self.dice_coeff(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_dice", dice_score, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True    )
        return {'optimizer': optimizer,'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'} }
    
    def dice_coeff(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.5).float()
        y = (y > 0.5).float()
        intersection = (y_hat * y).sum()
        return (2. * intersection) / (y_hat.sum() + y.sum() + 1e-6)
    
    def accuracy(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.5).float()
        y = (y > 0.5).float()
        correct = (y_hat == y).sum()
        return correct.float() / y.numel()

    def infer_image(self, image_generator, device='cuda'):
        self.eval()
        self.to(device)
        with torch.no_grad():
            for x, y in image_generator:
                x = x.to(device)                
                y_hat = self(x)  # Use self instead of model
                mask = (y_hat > 0.5).float()
                yield mask.squeeze(0).cpu().numpy()  # (C, H, W)

    def plot_predictions(self, loader, device='cuda'):
        self.to(device)
        self.eval()
        print("inicio de plot")
        images = []
        gif_frames = []
    
        with torch.no_grad():
            all_probs = []
            for x, y, id in loader:
                x = x.to(device)
                y = y.to(device).squeeze(1)  # Remove channel dimension
                y_hat = self(x)  # Use self instead of model
    
                x_np = x[:, 0].cpu().numpy()  
                y_np = y.cpu().numpy()  
                
                y_hat_np = (y_hat > 0.5).float()
                y_hat_np = y_hat_np.cpu().numpy()  # Ensure numpy conversion

                # 2. Histograma de probabilidades → se envía a WandB

                flat_probs = y_hat_np.flatten()
                fig_hist = plt.figure(figsize=(6, 4))
                plt.hist(flat_probs, bins=50, color='gray')
                plt.title("Distribución de valores predichos (sigmoid)")
                plt.xlabel("Probabilidad")
                plt.ylabel("Frecuencia")
                plt.grid(True)

                # 3. Log en WandB
                self.logger.experiment.log({"Sigmoid Output Histogram": wandb.Image(fig_hist)})
                plt.close(fig_hist)

                for i in range(len(x_np)):
                    images.append(wandb.Image(x_np[i], caption=f"Input - Sample {i}", mode="L"))
                    images.append(wandb.Image(y_np[i], caption=f"Ground Truth - Sample {i}", mode="L"))
                    images.append(wandb.Image(y_hat_np[i], caption=f"Prediction - Sample {i}", mode="L"))

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                    axes[0].imshow(x_np[i], cmap="gray")
                    axes[0].set_title(f"Input Image {id[i]}")
                    axes[0].axis("off")
    
                    axes[1].imshow(y_np[i], cmap="gray")
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis("off")

                    axes[2].imshow(y_hat_np[i], cmap="gray", vmin=0, vmax=1 )
                    axes[2].set_title(f"Prediction_sigmoid")
                    axes[2].axis("off")

                    buf = io.BytesIO()
                    plt.savefig(buf, format="jpg")
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
        train_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, target_size=p.RESIZE_VALUE, 
                                                     PATIENT_SPLITS = patient_splits, dataset_type="train",  block_id=p.BLOCK_ID)

        test_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, target_size=p.RESIZE_VALUE, 
                                                    PATIENT_SPLITS = patient_splits, dataset_type="test", block_id=p.BLOCK_ID)

        
        self.train_dataset = MainDataset(train_generator, augmentation=p.AUGMENTATION, dataset_type="Training")
        self.val_dataset = MainDataset(test_generator, augmentation=False, dataset_type="Test")
        self.val2_dataset = MainDataset(test_generator, augmentation=True, dataset_type="Test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=p.SHUFFLE, num_workers=p.WORKERS, persistent_workers=True, pin_memory=True, prefetch_factor=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=p.WORKERS, persistent_workers=True, pin_memory=True, prefetch_factor=4)

    def val2_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=p.WORKERS, persistent_workers=True, pin_memory=True, prefetch_factor=4)


torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="Lung-Segmentation", log_model=False)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=0)
    early_stop_callback = EarlyStopping(monitor ="val_dice", patience = round(p.EPOCHS*0.1, 0), verbose=True,mode="max")
    data_module = LungSegmentationDataModule()
    Callbacks = [checkpoint_callback]
    if p.EARLY_STOP:
        Callbacks.append(early_stop_callback)

    trainer = L.Trainer(logger=wandb_logger, callbacks=Callbacks, 
                        max_epochs=p.EPOCHS, accelerator="gpu", devices=2, 
                        precision="16-mixed", gradient_clip_val=1.0, 
                        enable_progress_bar=True, log_every_n_steps=20)

    if p.USE_PRETRAINED_MODEL:
        model = UNetLightning.load_from_checkpoint(os.path.join(p.RESULT_DIR, "model_final.ckpt"))
        print("Loaded pre-trained model.")

        if p.RE_TRAIN_MODEL:
            print("Re-training the pre-trained model.")
            trainer.fit(model, data_module)
        elif p.INFERENCE:
            inference_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, target_size=p.RESIZE_VALUE, dataset_type= "inference")
            model.infer_image(inference_generator)
        else:
            print("Testing pre-trained model performance.")
            trainer.validate(model, data_module)
    else:
        model = UNetLightning(lr=p.LEARNING_RATE)
        trainer.fit(model, data_module)
    
    if p.SAVE_PLOTS:
        model.plot_predictions(data_module.val2_dataloader())

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
