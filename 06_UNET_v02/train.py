import os
import time
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
            #cambiar por esta for i, (x, y) in enumerate(dl):
            #y por ende cambiar la funcion de resultado
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

def calculate_class_weights(mask):

    mask_flat = mask.flatten()
    

    total_pixels = mask_flat.numel()
    foreground_pixels = mask_flat.sum().item()
    background_pixels = total_pixels - foreground_pixels
    
    background_weight = foreground_pixels / total_pixels
    foreground_weight = background_pixels / total_pixels
    
    return torch.tensor([background_weight, foreground_weight])

def train_model(dl, test_dl, model, device, n_epochs):
    # Optimization
    opt = Adam(model.parameters(), lr=3e-4)  # karpathy's constant
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss with logits

    # Initialize progress bar and timing
    start_time = time.time()
    total_steps = n_epochs * len(dl)

    # Train model
    epochs = []
    accuracy = []
    losses = []
    test_losses = []
    test_accuracy = []

    with tqdm(total=total_steps, dynamic_ncols=True, leave=True) as pbar:
        for epoch in range(n_epochs):
            model.train()
            N = len(dl)

            for i, (x, y) in enumerate(dl):
                x, y = x.to(device), y.to(device).squeeze(1)#.long()  # Convert mask to [B, H, W] and integer type
                opt.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                opt.step()

                # Store training data
                epochs.append(epoch + i / N)
                training_loss += loss.item()

                # Update progress bar
                pbar.update(1)
            
            losses.append(training_loss/ len(dl))

            # Evaluate model on test set
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(dl):
                    x, y = x.to(device), y.to(device).squeeze(1)
                    test_outputs = model(x)
                    test_loss += criterion(test_outputs, y).item()
            
            test_losses.append(test_loss/len(test_dl))

            accuracy.append(CheckAccuracy(dl, model, device))
            test_accuracy.append(CheckAccuracy(test_dl, model, device)) ###cehck accurassy testing data


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
                "Last Batch Loss": f"{loss.item():.4f}",
                "Accurasy": f"{accuracy[-1]:.4f}",
                "Estimated Time Left": f"{int(estimated_time_left // 60)}m {int(estimated_time_left % 60)}s"
            })
        if p.SAVE_MODEL:
            model_path = os.path.join(p.RESULT_DIR, 'modelo_UNET_1.pth')
            torch.save(model.state_dict(), model_path)

    return np.array(epochs), np.array(losses), np.array(test_losses), accuracy, test_accuracy