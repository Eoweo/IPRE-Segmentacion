import os
import matplotlib.pyplot as plt
import torch

class Report:
    def __init__(self, epoch, loss, Accuracy, RESULT_DIR = "Result", n_epochs=20):
        self.epoch = epoch
        self.loss = loss
        self.type = type
        self.result_dir = RESULT_DIR
        self.n_epochs = n_epochs
        self.Accuracy = torch.tensor(Accuracy, dtype=torch.float32).reshape(-1)
        self.epoch_data_avgd = self.epoch  # Adjust as needed for averaging
        self.loss_data_avgd = self.loss   # Adjust as needed for averaging

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
            y_hat = torch.argmax(y_hat, dim=1).cpu().numpy()  # Convert logits to class labels

            # Store data for visualization
            inputs.extend(x[:, 0].cpu().numpy())
            ground_truths.extend(y.cpu().numpy())
            predictions.extend(y_hat)

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
