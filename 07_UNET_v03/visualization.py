import os
import io
import gc
import wandb
import psutil
import torch
import imageio
import parameter as p
from model import UNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import train_model, CheckAccuracy
from dataset import tif_dataset_generator  , MainDataset, initialize_patient_splits, load_jpg_dataset_generator


def plot_predictions_interactive(model, loader, device='cuda'):
    model.eval()
    wandb_images = []
    gif_frames = [] 

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).squeeze(1)  # Remove channel dimension

            y_hat = model(x)

            x_np = x[:, 0].cpu().numpy()  
            y_np = y.cpu().numpy()  
            y_hat_np = y_hat.cpu()

            for i in range(len(x_np)):
                wandb_images.append(
                    wandb.Image(
                        x_np[i],
                        caption=f"Input - Sample {i}",
                        mode="L"
                    ))
                wandb_images.append(
                    wandb.Image(
                        y_np[i],
                        caption=f"Ground Truth - Sample {i}",
                        mode="L"
                    ))
                wandb_images.append(
                    wandb.Image(
                        y_hat_np[i],
                        caption=f"Prediction - Sample {i}",
                        mode="L"
                    ))
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(x_np[i], cmap="gray")
                axes[0].set_title("Input Image")
                axes[0].axis("off")

                axes[1].imshow(y_np[i], cmap="gray")
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                axes[2].imshow(y_hat_np[i], cmap="gray")
                axes[2].set_title(f"Prediction")
                axes[2].axis("off")

                # Convert to an in-memory image
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                gif_frames.append(imageio.imread(buf))  # Add to GIF

    wandb.log({"Segmentation Results": wandb_images})

    gif_path = "predictions_over_epochs.gif"
    imageio.mimsave(gif_path, gif_frames, duration=0.5)  # 0.5s per frame
    wandb.log({"Predictions GIF": wandb.Video(gif_path, fps=2, format="gif")})




def Menu():
    print("Initializing Menu...")
    
    #|-----------------SELECT DATASET------------------------------|

    dataset_choice = p.TEST_AVAILABLE[p.TEST_SELECTED_INDEX]
    
    if dataset_choice == "EPFL - Mitocondria Electron Microscopy":
        test_generator = tif_dataset_generator(p.PATH_EPFL, "test")
        train_generator = tif_dataset_generator(p.PATH_EPFL, "train")

    elif dataset_choice == "Chest CT Segmentation":
        print("set dataset")
        PATIENT_SPLITS = initialize_patient_splits(p.PATH_CT_MARCOPOLO)
        test_generator =  load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, PATIENT_SPLITS, dataset_type="test" , target_size = p.RESIZE_VALUE, block_id=p.BLOCK_ID)
        train_generator = load_jpg_dataset_generator(p.PATH_CT_MARCOPOLO, PATIENT_SPLITS, dataset_type="train", target_size = p.RESIZE_VALUE, block_id=p.BLOCK_ID)
    
    #|-------------PREPARE DATASET AND DATALOADER-----------------|

    print("set MainDataset")
    if p.RE_TRAIN_MODEL or not p.USE_PRETRAINED_MODEL:
        train_dataset = MainDataset(train_generator, p.AUGMENTATION, "Training")
    test_dataset = MainDataset(test_generator, p.AUGMENTATION, "Test")
    
    print("Set Dataloader")
    if p.RE_TRAIN_MODEL or not p.USE_PRETRAINED_MODEL:
        train_dl = DataLoader(train_dataset, batch_size=p.BATCH_SIZE, shuffle=p.SHUFFLE, pin_memory=True, num_workers=19)
    test_dl = DataLoader(test_dataset, batch_size=p.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=19)
    
    #|--------------CHECK AND CLEAN MEMORY USAGE-------------------|

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

    #|--------------INITIALIZE THE MODEL-------------------|
    
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
                print("-------------- Saving plots --------------")
                plot_predictions_interactive(model, test_dl, device=device)
    
    #|--------------TRAIN-------------------|

    if p.RE_TRAIN_MODEL or not p.USE_PRETRAINED_MODEL:
        # Train the model
        initialize_patient_splits(p.PATH_CT_MARCOPOLO)
        train_model(train_dl, test_dl, model, device)
        accuracy = CheckAccuracy(test_dl, model, device)
    
        # Save model if enabled
        if p.SAVE_MODEL:
            model_path = os.path.join(p.RESULT_DIR, 'modelo_UNET_1.pth')
            torch.save(model.state_dict(), model_path)
        
        # Save and visualize results if enabled
        if p.SAVE_PLOTS:
            plot_predictions_interactive(model, train_dl, device=device)
        print(f"Test accuracy obtained: {accuracy:.4f}")


