import os
import matplotlib.pyplot as plt
import torch
import parameter as p
from dataset import set_tif_dataset, set_jpg_Dataset, MainDataset
from train import train_model, CheckAccuracy
from model import UNet
from torch.utils.data import DataLoader

class Report:
    def __init__(self, epoch, loss, Accuracy, n_epochs, RESULT_DIR = "Result"):
        self.epoch = epoch
        self.loss = loss
        self.type = "Training"
        self.result_dir = RESULT_DIR
        self.n_epochs = n_epochs
        self.Accuracy = torch.tensor(Accuracy, dtype=torch.float32).reshape(-1)
        self.epoch_data_avgd = self.epoch.reshape(self.n_epochs,-1).mean(axis=1)
        self.loss_data_avgd = self.loss.reshape(self.n_epochs,-1).mean(axis=1)

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Plot Loss
        axes[0].plot(self.epoch_data_avgd, self.loss_data_avgd, 'o--', label='Loss', color="cyan")
        axes[0].set_xlabel('Epoch Number')
        axes[0].set_ylabel('Cross Entropy')
        axes[0].set_title(f'Cross Entropy (avgd per epoch) - {self.type}')
        axes[0].legend()

        # Plot Accuracy
        axes[1].plot(self.epoch_data_avgd, self.Accuracy, 'd--', label='Accuracy', color='orange')
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
    while True:
        print("\n==== UNET Neural Network ====")
        print("""
        [1] Choose Test Dataset
        [2] View or Modify Parameters
        [3] Run Neural Network
        [4] Exit Program
        """)
        option = input("Please select an option (1, 2, 3, or 4): ").strip()

        if option == "1":
            # Choose a test dataset
            print("\nAvailable Test Datasets:")
            for i, test in enumerate(p.TEST_AVAILABLE, start=1):
                print(f"[{i}] {test}")
            selected = int(input("Choose a test dataset (e.g., 1 or 2): ").strip())
            dataset_choice = p.TEST_AVAILABLE[selected - 1]
            print(f"You selected: {dataset_choice}")

        elif option == "2":
            # View or modify parameters
            print("\nCurrent Parameters:")
            for attr in dir(p):
                if not attr.startswith("__"):
                    print(f"{attr}: {getattr(p, attr)}")

            while True:
                change = input("\nWould you like to change a parameter? (yes/no): ").strip().lower()
                if change in "yes":
                    param_to_change = input("Enter the name of the parameter to change: ").strip().upper()
                    if hasattr(p, param_to_change):
                        new_value = input(f"Enter a new value for {param_to_change}: ").strip()
                        try:
                            # Update the parameter
                            value = eval(new_value)  # Convert string to the appropriate data type
                            setattr(p, param_to_change, value)
                            print(f"Updated {param_to_change} to {value}")
                        except Exception as e:
                            print(f"Error updating parameter: {e}")
                    else:
                        print("Invalid parameter name.")
                elif change in "no":
                    print("Exiting parameter modification.")
                    break
                else:
                    print("Invalid input. Please type 'yes' or 'no'.")

        elif option == "3":
            # Run the neural network
            save_model = input("Do you want to save the trained model? (yes/no): ").strip().lower() in "yes"
            save_plots = input("Do you want to save the prediction plots? (yes/no): ").strip().lower() in "yes"

            # Load the dataset
            if dataset_choice == "EPFL - Mitocondria Electron Microscopy":
                train_ds, train_mask_ds, test_ds, test_mask_ds = set_tif_dataset('Database\\EPFL')
            elif dataset_choice == "Chest CT Segmentation":
                train_ds, train_mask_ds, test_ds, test_mask_ds = set_jpg_Dataset('Database\\CT-Chest\\Marco Polo\\archive')
            else:
                print("Invalid dataset choice.")
                continue

            # Prepare datasets and dataloaders
            train_dataset = MainDataset(train_ds[:p.CHOP_VALUE], train_mask_ds[:p.CHOP_VALUE], p.ROTATION)
            test_dataset = MainDataset(test_ds, test_mask_ds, False)

            train_dl = DataLoader(train_dataset, batch_size=p.BATCH_SIZE, shuffle=p.SHUFFLE, pin_memory=True)
            test_dl = DataLoader(test_dataset, batch_size=p.BATCH_SIZE, shuffle=False, pin_memory=True)

            # Initialize the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UNet(in_channels=1, out_channels=1).to(device)

            # Train the model
            epoch_data, loss_data, Accurasy_data = train_model(train_dl, model, device, n_epochs=p.EPOCHS)
            
            acurracy = CheckAccuracy(test_dl, model, device)

            if save_model:
                model_path = os.path.join(p.RESULT_DIR, 'modelo_UNET_1.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")

            # Visualize predictions
            if save_plots:
                Analisis = Report(epoch_data, loss_data, Accurasy_data,  p.EPOCHS)
                Analisis.plot()
                plot_predictions_interactive(model, test_dl, device=device)
            print((f"Test accuracy obtain: {acurracy:.4f}"))

        elif option == "4":
            print("Exiting the program.")
            break
        elif option == "5":

            if dataset_choice == "EPFL - Mitocondria Electron Microscopy":
                train_ds, train_mask_ds, test_ds, test_mask_ds = set_tif_dataset('Database\\EPFL')
            elif dataset_choice == "Chest CT Segmentation":
                train_ds, train_mask_ds, test_ds, test_mask_ds = set_jpg_Dataset('Database\\CT-Chest\\Marco Polo\\archive')
            else:
                print("Invalid dataset choice.")
                continue
            
            input("Please place the file named as model.pth in the same folder as main.py and then press ENTER")
            save_plots = input("Do you want to save the prediction plots? (yes/no): ").strip().lower() in "yes"

            test_dataset = MainDataset(test_ds, test_mask_ds, False)
            test_dl = DataLoader(test_dataset, batch_size=p.BATCH_SIZE, shuffle=False, pin_memory=True)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UNet(in_channels=1, out_channels=1).to(device)
            model_path = "model.pth"
            model.load_state_dict(torch.load(model_path, weights_only=True))
            if save_plots:
                plot_predictions_interactive(model, test_dl, device=device)
            acurracy = CheckAccuracy(test_dl, model, device)
            print(input(f"Test accuracy obtain: {acurracy}"))

        else:
            print("Invalid option. Please try again.")
