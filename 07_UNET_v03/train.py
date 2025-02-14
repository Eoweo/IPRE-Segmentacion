import os
import time
import wandb
import psutil
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
import torch.nn as nn
import parameter as p

def CheckAccuracy(loader, model, device):
    dice_scores = []
    model.eval()

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
            
            # Calculate Dice Score
            intersection = (y_hat * y).sum(dim=(1, 2))
            union = y_hat.sum(dim=(1, 2)) + y.sum(dim=(1, 2))
            dice = (2.0 * intersection) / (union + 1e-12 )  # Add epsilon to avoid division by zero
            dice_scores.append(dice.mean().item())

    model.train()
    return sum(dice_scores)/len(dice_scores)

def train_model(dl, test_dl, model, device):

    opt = Adam(model.parameters(), lr=3e-4)  # karpathy's constant
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss with logits

    n_epochs = wandb.config.epochs
    best_loss = float("inf")
    patience_counter = 0
    start_time = time.time()
    total_steps = n_epochs * len(dl)

    # Train model
    epochs = []

    with tqdm(total=total_steps, dynamic_ncols=True, leave=True) as pbar:
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0

            for x, y in dl:

                x, y = x.to(device), y.to(device).squeeze(1)#.long()  # Convert mask to [B, H, W] and integer type
                opt.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

                pbar.update(1)

            epochs.append(epoch)
            train_losses = epoch_loss/ len(dl)

            # Evaluate model on test set
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for x, y in dl:
                    x, y = x.to(device), y.to(device).squeeze(1)
                    test_outputs = model(x)
                    test_loss += criterion(test_outputs, y).item()
            
            test_losses = test_loss/len(test_dl)
            
            train_accuracy = CheckAccuracy(dl, model, device)
            test_accuracy = CheckAccuracy(test_dl, model, device)

            wandb.log({
                "Accuracy": {
                    "Train Accuracy": train_accuracy,
                    "Test Accuracy": test_accuracy
                },
                "Loss": {
                    "Train Loss": train_losses,
                    "Test Loss": test_losses
                }
            })

            # Calculate time left
            elapsed_time = time.time() - start_time
            remaining_steps = total_steps - pbar.n
            time_per_step = elapsed_time / pbar.n if pbar.n > 0 else 0
            estimated_time_left = time_per_step * remaining_steps
            used_memory =  psutil.virtual_memory().used / (1024**3)
            total_memory = psutil.virtual_memory().total / (1024**3)

            # Set progress bar postfix with estimated time left
            pbar.set_postfix({
                "Mem": f"{used_memory:.2f} / {total_memory:.2f} GB",
                "Epoch": f"{epoch + 1}/{n_epochs}",
                "train_L": f"{train_losses:.4f}",
                "Test_L": f"{test_losses:.4f}",                
                "Train_Acc": f"{train_accuracy:.4f}",
                "Test_Acc": f"{test_accuracy:.4f}",                
                "Aprox. Time Left": f"{int(estimated_time_left // 60)}m {int(estimated_time_left % 60)}s"
            })
            patience = 10 #n_epochs*0.05
            if test_losses < best_loss:
                best_loss = test_losses
                patience_counter = 0
                best_model_state = model.state_dict() 
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs.")

            if patience_counter >= patience:
                print(f"Stopping early! No improvement for {patience} epochs.")

                torch.save(best_model_state, "best_model.pth")
                wandb.save("best_model.pth")

                break  

    return

