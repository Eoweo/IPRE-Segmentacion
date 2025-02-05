import os
import gc
import psutil
import matplotlib.pyplot as plt
import torch
import parameter as p
from dataset import set_tif_dataset, MainDataset, initialize_patient_splits, load_jpg_dataset_generator
from train import train_model, CheckAccuracy
from model import UNet
from torch.utils.data import DataLoader
import kagglehub


class Report:
    def __init__(self, epoch, loss, test_losses, Accuracy, test_accuracy, n_epochs, RESULT_DIR = "Result"):
        self.epoch = [i for i in range(epoch)]
        self.loss = loss
        self.type = ["Training", "Test"]
        self.result_dir = RESULT_DIR
        self.n_epochs = n_epochs
        self.Accuracy = [torch.tensor(Accuracy, dtype=torch.float32).reshape(-1),
                         torch.tensor(test_accuracy, dtype=torch.float32).reshape(-1)]
        self.loss_data_avgd = [torch.tensor(loss, dtype=torch.float32).reshape(-1),
                               torch.tensor(test_losses, dtype=torch.float32).reshape(-1)]
        
        self.epoch_data_avgd = self.epoch.reshape(self.n_epochs,-1).mean(axis=1)

    def plot(self):
        for i in range(2):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Plot Loss
            axes[0].plot(self.epoch, self.loss_data_avgd, 'o--', label='Loss', color="cyan")
            axes[0].set_xlabel('Epoch Number')
            axes[0].set_ylabel('Cross Entropy')
            axes[0].set_title(f'Cross Entropy (avgd per epoch) - {self.type}')
            axes[0].legend()

            # Plot Accuracy
            axes[1].plot(self.epoch, self.Accuracy, 'd--', label='Accuracy', color='orange')
            axes[1].set_xlabel('Epoch Number')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_xlim(0, self.n_epochs)
            axes[1].set_title(f'Accuracy Over Epochs - {self.type}')
            axes[1].legend()

            plt.tight_layout()
            save_path = os.path.join(self.result_dir, f'{self.type}_Report.png')
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
            plt.close()  # Close the plot to avoid memory issues

def plot_predictions_interactive(model, loader, RESULT_DIR = "Result", device='cuda'):
    model.eval()
    inputs, ground_truths, predictions = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).squeeze(1)  # Remove channel dimension

            # Model prediction
            y_hat = model(x)

            if len(y_hat.shape) == 4 and y_hat.shape[1] > 1:  # Multi-class output
                y_hat = torch.argmax(y_hat, dim=1)  # Convert logits to class labels
            elif len(y_hat.shape) == 4 and y_hat.shape[1] == 1:  # Binary segmentation
                y_hat = (y_hat > 0).float().squeeze(1)  # Threshold logits at 0.0

            # Store data for visualization
            inputs.extend(x[:, 0].cpu().numpy())
            ground_truths.extend(y.cpu().numpy())
            predictions.extend(y_hat.cpu())

    def save_sample(idx):
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        axes[0].imshow(inputs[idx], cmap='gray')
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        axes[1].imshow(ground_truths[idx], cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

        axes[2].imshow(predictions[idx], cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        save_path = os.path.join(RESULT_DIR, f'Prediction_{idx}.png')
        plt.savefig(save_path)
        print(f"Saved prediction to {save_path}")
        plt.close()  # Close the plot to avoid memory issues

    # Save all plots
    for idx in range(len(inputs)):
        save_sample(idx)

def Menu():
    print("Initializing Menu...")
    
    # Select test dataset
    dataset_choice = p.TEST_AVAILABLE[p.TEST_SELECTED_INDEX]  # Use predefined index
    
    if dataset_choice == "EPFL - Mitocondria Electron Microscopy":
        train_ds, train_mask_ds, test_ds, test_mask_ds = set_tif_dataset(p.PATH_EPFL)
    elif dataset_choice == "Chest CT Segmentation":
        print("set dataset")
        initialize_patient_splits(p.PATH_CT_MARCOPOLO)
        test_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, dataset_type="test", block_id=p.BLOCK_ID)
        train_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, dataset_type="train", block_id=p.BLOCK_ID)
    
    # Prepare datasets and dataloaders
    print("set MainDataset")
    if p.RE_TRAIN_MODEL or not p.USE_PRETRAINED_MODEL:
        train_dataset = MainDataset(train_generator, p.ROTATION, type="Training")
    test_dataset = MainDataset(test_generator, False)
    
    print("Set Dataloader")
    if p.RE_TRAIN_MODEL or not p.USE_PRETRAINED_MODEL:
        train_dl = DataLoader(train_dataset, batch_size=p.BATCH_SIZE, shuffle=p.SHUFFLE, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=p.BATCH_SIZE, shuffle=False, pin_memory=True, )
    
    used_memory =  psutil.virtual_memory().used / (1024**3)
    total_memory = psutil.virtual_memory().total / (1024**3)
    print(f"actual memory{used_memory:.2f} / {total_memory:.2f} GB")
    if p.RE_TRAIN_MODEL or not p.USE_PRETRAINED_MODEL:
        del train_dataset
    del test_dataset
    gc.collect()
    used_memory =  psutil.virtual_memory().used / (1024**3)
    total_memory = psutil.virtual_memory().total / (1024**3)
    print(f"clean memory{used_memory:.2f} / {total_memory:.2f} GB")



    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    if p.USE_PRETRAINED_MODEL:
        model_path = os.path.join(p.RESULT_DIR, "..", "model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Using pre-trained model.")
        
        if not p.RE_TRAIN_MODEL:
            accuracy = CheckAccuracy(test_dl, model, device)
            print(f"Test accuracy obtained: {accuracy:.4f}")

            if p.SAVE_PLOTS:
                plot_predictions_interactive(model, test_dl, device=device, RESULT_DIR=p.RESULT_DIR)
    
    if p.RE_TRAIN_MODEL or not p.USE_PRETRAINED_MODEL:
        # Train the model
        initialize_patient_splits(p.PATH_CT_MARCOPOLO)
        epoch_data, loss_data, accuracy_data, accuracy_test = train_model(train_dl, test_dl, model, device, n_epochs=p.EPOCHS)
        accuracy = CheckAccuracy(test_dl, model, device)
    
        # Save model if enabled
        if p.SAVE_MODEL:
            model_path = os.path.join(p.RESULT_DIR, 'modelo_UNET_1.pth')
            torch.save(model.state_dict(), model_path)
    
        # Save and visualize results if enabled
        if p.SAVE_PLOTS:
            analysis = Report(epoch_data, loss_data, accuracy_data, accuracy_test, p.EPOCHS, RESULT_DIR=p.RESULT_DIR)
            analysis.plot()
            plot_predictions_interactive(model, test_dl, device=device, RESULT_DIR=p.RESULT_DIR)
    
        print(f"Test accuracy obtained: {accuracy:.4f}")


