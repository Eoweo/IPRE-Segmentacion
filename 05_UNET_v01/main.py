import os
import sys
import torch
from torch.utils.data import DataLoader
from model import UNet
from dataset import set_tif_dataset, set_jpg_Dataset, MainDataset
from train import train_model
from visualization import plot_predictions_interactive


if __name__ == "__main__":

    entrada = sys.argv

    RESULT_DIR = "Result"
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    mitocondría = True
    if mitocondría:
        train_ds, train_mask_ds, test_ds, test_mask_ds = set_tif_dataset('..\\Database\\EPFL')

    train_dataset = MainDataset(train_ds[:50], train_mask_ds[:50])
    test_dataset = MainDataset(test_ds, test_mask_ds)

    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=4, shuffle=True, pin_memory=True)

    # Initialize model
    model = UNet(in_channels=1, out_channels=2).to(device)

    # Train model
    epoch_data, loss_data, Accurasy_data = train_model(train_dl, model, n_epochs=10, device='cuda')

    # Save model
    
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'modelo_UNET_1.pth'))

    # Visualize predictions
    plot_predictions_interactive(model, test_dl, device=device)