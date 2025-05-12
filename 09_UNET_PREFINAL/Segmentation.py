import lightning.pytorch as L
import io
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
import seaborn as sns
import imageio.v2 as imageio
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
import wandb
from model import UNet
from dataset import MainDataset, load_jpg_dataset_generator, initialize_patient_splits
import parameter as p

class UNetLightning(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, lr=p.LEARNING_RATE):
        super(UNetLightning, self).__init__()
        self.save_hyperparameters("lr", "in_channels", "out_channels")
        self.lr = lr
        self.model = UNet(in_channels, out_channels)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x, y, id = batch
        x, y = x, y.squeeze(1)
        y = (y > 0.5).float()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        dice_score = self.dice_coeff(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True, batch_size=p.BATCH_SIZE)
        self.log("train_dice", dice_score, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True, batch_size=p.BATCH_SIZE)
        self.log("train_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True, batch_size=p.BATCH_SIZE)
        self.log("epoch", self.current_epoch, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True, batch_size=p.BATCH_SIZE)
        return loss
    
    def validation_step(self, batch):
        x, y, id = batch
        x, y = x, y.squeeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        dice_score = self.dice_coeff(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, batch_size=p.BATCH_SIZE)
        self.log("val_dice", dice_score, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True, batch_size=p.BATCH_SIZE)
        self.log("val_accuracy", accuracy, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True, batch_size=p.BATCH_SIZE)
        self.log("epoch", self.current_epoch, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True, batch_size=p.BATCH_SIZE)
        return loss

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        return {'optimizer': optimizer,'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'} }

    def dice_coeff(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.5).float()
        y = (y > 0.5).float()
        intersection = (y_hat * y).sum()
        denominator = y_hat.sum() + y.sum()
        
        return ((2. * intersection) / (y_hat.sum() + y.sum() + 1e-6)) + (denominator == 0).float() #add border case when perfect match but black images
    
    def accuracy(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.5).float()
        y = (y > 0.5).float()
        correct = (y_hat == y).sum()
        return correct.float() / y.numel()
    
    def on_train_start(self):
        print(f"Devices used for training: {self.trainer.num_devices}")
        print(f"Accelerator: {self.trainer.accelerator}")
        print(f"Strategy: {self.trainer.strategy}")

    @rank_zero_only
    def plot_predictions(self, loader, device='cuda'):
        self.to(device)
        self.eval()
        print("inicio de plot")
        images = []
        gif_frames = []
        dice_scores = []
        accuracies = []

        with torch.no_grad():
            step_i = 0
            for x, y, id in loader:
                x = x.to(device)
                y = y.to(device).squeeze(1)  # Remove channel dimension
                y_hat = self(x)  # Use self instead of model
    
                x_np = x[:, 0].cpu().numpy()  
                y_np = y.cpu().numpy()  
                
                y_hat_np = (y_hat > 0.5).float()
                y_hat_np = y_hat_np.cpu().numpy()  # Ensure numpy conversion
                
                for i in range(len(x_np)):
                    pred_i = torch.tensor(y_hat_np[i], dtype=torch.float32).unsqueeze(0)
                    true_i = torch.tensor(y_np[i], dtype=torch.float32).unsqueeze(0)

                    dice_score = self.dice_coeff(pred_i, true_i)
                    accuracy = self.accuracy(pred_i, true_i)
                    dice_scores.append(dice_score)
                    accuracies.append(accuracy)
                    wandb.log({
                        "test_dice": dice_score,
                        "test_accuracy": accuracy,
                        "test_step": step_i})
                    step_i = step_i + 1
                    images.append(wandb.Image(x_np[i], caption=f"Input {i}", mode="L"))
                    images.append(wandb.Image(y_np[i], caption=f"Ground Truth {i}", mode="L"))
                    images.append(wandb.Image(y_hat_np[i], caption=f"Prediction - Dice: {dice_score:.3f} | Acc: {accuracy:.3f}"))

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                    axes[0].imshow(x_np[i], cmap="gray")
                    axes[0].set_title(f"Test Image {id[i]}")
                    axes[0].axis("off")
    
                    axes[1].imshow(y_np[i], cmap="gray")
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis("off")

                    axes[2].imshow(y_hat_np[i], cmap="gray", vmin=0, vmax=1 )
                    axes[2].set_title(f"Test Inference - Dice: {dice_score:.3f} | Acc: {accuracy:.3f}")
                    axes[2].axis("off")

                    buf = io.BytesIO()
                    plt.savefig(buf, format="jpg")
                    plt.close()
                    buf.seek(0)
                    gif_frames.append(imageio.imread(buf))
        
        avg_dice = np.mean(dice_scores)
        avg_accuracy = np.mean(accuracies)

        wandb.log({
            "avg_test_dice": avg_dice,
            "avg_test_accuracy": avg_accuracy})
        
        self.logger.experiment.log({"Segmentation Results": images})
        gif_name = get_name(base_name = "predictions",folder=p.RESULT_DIR, ext=".gif")
        gif_path = os.path.join(p.RESULT_DIR, gif_name)
        imageio.mimsave(gif_path, gif_frames, duration=0.5)
        self.logger.experiment.log({"Predictions GIF": wandb.Video(gif_path, fps=2, format="gif")})
        print(f"gif saved and uploaded in {gif_path}.")

class LungSegmentationDataModule(L.LightningDataModule):
    def __init__(self, batch_size=p.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                initialize_patient_splits(p.PATH_CT_MARCOPOLO)
            dist.barrier()
    
        patient_splits = initialize_patient_splits(p.PATH_CT_MARCOPOLO)

        train_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, target_size=p.RESIZE_VALUE, 
                                                     PATIENT_SPLITS = patient_splits, dataset_type="train",  block_id=p.BLOCK_ID)
        
        validation_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, target_size=p.RESIZE_VALUE, 
                                                    PATIENT_SPLITS = patient_splits, dataset_type="validation", block_id=p.BLOCK_ID)
        
        test_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, target_size=p.RESIZE_VALUE, 
                                                    PATIENT_SPLITS = patient_splits, dataset_type="test", block_id=p.BLOCK_ID)
        
        self.train_dataset = MainDataset(train_generator, augmentation=p.AUGMENTATION, dataset_type="train")
        self.val_dataset = MainDataset(validation_generator, augmentation=False, dataset_type="validation")
        self.test_dataset = MainDataset(test_generator, augmentation=False, dataset_type="test")
        print("Validation dataset length:", len(self.val_dataset))
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=p.WORKERS, 
                          persistent_workers=True, pin_memory=True, prefetch_factor=2, drop_last=True)
    
    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=True)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=p.WORKERS, 
                          persistent_workers=True, pin_memory=True, prefetch_factor=2, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=p.WORKERS, 
                          persistent_workers=True, pin_memory=True, prefetch_factor=2)

class CleanOldCheckpoints(Callback):
    def __init__(self, keep=2):
        super().__init__()
        self.keep = keep

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        ckpt_dir = trainer.checkpoint_callback.dirpath
        if ckpt_dir is None or not os.path.exists(ckpt_dir):
            return  # Evita error si aún no se ha creado el directorio

        all_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")])
        if len(all_ckpts) > self.keep:
            to_delete = all_ckpts[:-self.keep]
            for f in to_delete:
                path = os.path.join(ckpt_dir, f)
                try:
                    os.remove(path)
                    print(f"Deleted old checkpoint: {f}")
                except Exception as e:
                    print(f"Failed to delete {f}: {e}")

def get_name(base_name, folder, ext=".ckpt"):
    version = 0
    while True:
        name = f"{base_name}_{version:02d}{ext}"
        if not os.path.exists(os.path.join(folder, name)):
            return os.path.join(p.RESULT_DIR, name)
        version += 1

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    #Delete json
    @rank_zero_only
    def remove_patient_splits():
        if os.path.exists("patient_splits.json"):
            os.remove("patient_splits.json")
    remove_patient_splits()
    #print to the actual version
    nombre_archivo = "/home/cmorenor/Eoweo/IPRE-Segmentacion/09_UNET_PREFINAL/Segmentation.py"
    with open(nombre_archivo, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()

    print(contenido)

    L.seed_everything(1)
    wandb_logger = WandbLogger(project="Lung-Segmentation_Final", log_model=True, save_dir = p.RESULT_DIR)
    log_params = {
    "augmentation": p.AUGMENTATION,
    "resize": p.RESIZE,
    "resize_value": p.RESIZE_VALUE,
    "ratio": p.RATIO,
    "chop_data": p.CHOP_DATA,
    "chop_patient": p.CHOP_PATIENT,
    "epochs": p.EPOCHS,
    "batch_size": p.BATCH_SIZE,
    "learning_rate": p.LEARNING_RATE,
    "shuffle": p.SHUFFLE,
    "workers": p.WORKERS,
    "multi_class": p.MULTI_CLASS,
    "save_model": p.SAVE_MODEL,
    "save_plots": p.SAVE_PLOTS,
    "early_stop": p.EARLY_STOP,
    "use_pretrained": p.USE_PRETRAINED_MODEL,
    "re_train_model": p.RE_TRAIN_MODEL,
    "block_id_count": len(p.BLOCK_ID),
    "block_ids": p.BLOCK_ID}

    print("Configuración del experimento:")
    for k, v in log_params.items():
        print(f"  {k}: {v}")
    # Log a Weights & Biases
    wandb_logger.log_hyperparams(log_params)

    data_module = LungSegmentationDataModule()
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,save_last=True,every_n_epochs=10)
    early_stop_callback = EarlyStopping(monitor ="val_dice", patience = round(p.EPOCHS*0.2, 0), verbose=True,mode="max")
    Callbacks = [checkpoint_callback]#, CleanOldCheckpoints(keep=2)]
    if p.EARLY_STOP:
        Callbacks.append(early_stop_callback)

    trainer = L.Trainer(logger=wandb_logger, callbacks=Callbacks, profiler="advanced",
                        max_epochs=p.EPOCHS, accelerator="gpu", devices=-1, 
                        precision="16-mixed", gradient_clip_val=1.0, 
                        enable_progress_bar=True, log_every_n_steps=20, 
                        strategy='ddp' )

    if p.USE_PRETRAINED_MODEL:
        model = UNetLightning.load_from_checkpoint(os.path.join(p.RESULT_DIR, "model.ckpt"))
        print("Loaded pre-trained model.")

        if p.RE_TRAIN_MODEL:
            print("Re-training the pre-trained model.")
            trainer.fit(model, data_module)
        else:
            print("Testing pre-trained model performance.")
            trainer.validate(model, data_module)
    else:
        model = UNetLightning(lr=p.LEARNING_RATE)
        trainer.fit(model, data_module)
    
    if trainer.global_rank == 0:
        if p.SAVE_PLOTS:
            model.plot_predictions(data_module.test_dataloader())
        if p.SAVE_MODEL:
            # Save .pth
            model_name_1 = get_name(base_name="model", folder=p.RESULT_DIR, ext=".pth")
            torch.save(model.model.state_dict(), model_name_1)
            print(f"Model saved and uploaded in {model_name_1}.")

            # Save .ckpt
            #model_name_2 = get_name(base_name="model", folder=p.RESULT_DIR, ext=".ckpt")
            #trainer.save_checkpoint(model_name_2)
            #print(f"Model saved and uploaded in {model_name_2}.")

            ## Upload both to W&B as artifacts
            #artifact_pth = wandb.Artifact(name="modelos/model_pth", type="model")
            #artifact_pth.add_file(model_name_1)
            #wandb_logger.experiment.log_artifact(artifact_pth)
#
            #artifact_ckpt = wandb.Artifact(name="modelos/model_ckpt", type="model")
            #artifact_ckpt.add_file(model_name_2)
            #wandb_logger.experiment.log_artifact(artifact_ckpt)
#
            #print(f"Model saved locally and uploaded in folder 'modelos' on W&B.")

            #print(f"Model saved and uploaded in {model_name_1} and {model_name_2}.")
        #if p.SAVE_MODEL:
        #    best_path = checkpoint_callback.best_model_path
        #    model = UNetLightning.load_from_checkpoint(best_path)
        #    model_name_1 = get_name(base_name = "model",folder=p.RESULT_DIR, ext=".pth")
        #    checkpoint = torch.save(model.model.state_dict(), model_name_1)
        #    wandb_logger.experiment.save("Model": checkpoint)
        #    print(f"Model saved and uploaded in {model_name_1}.")
    wandb.finish()