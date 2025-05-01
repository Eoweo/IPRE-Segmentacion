import os
import torch
from PIL import Image
import numpy as np
from model import UNet 
import parameter as p  

model_path = os.path.join(p.RESULT_DIR, "model.pth")
NUM_CLASSES = 1  # 1 for binary, >1 for multiclass

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(p.RESULT_DIR, exist_ok=True)

# Load model
model = UNet(in_channels=1, out_channels=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Inference
with torch.no_grad():
    for filename in os.listdir(p.INFERENCE_PATH):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(p.INFERENCE_PATH, filename)
        image = Image.open(img_path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0
        print(image.shape)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # [batch_size, channels, height, width]

        y_hat = model(image)

        if NUM_CLASSES == 1:
            pred_mask = (torch.sigmoid(y_hat) > 0.5).float()
        else:
            pred_mask = torch.argmax(torch.softmax(y_hat, dim=1), dim=1, keepdim=True).float()
        
        print(y_hat.shape)
        pred_mask_np = pred_mask.squeeze().cpu().numpy() * 255
        pred_mask_img = Image.fromarray(pred_mask_np.astype(np.uint8))
        pred_mask_img.save(os.path.join(p.RESULT_DIR, f"{os.path.splitext(filename)[0]}_mask.png"))

print(f"Inference done. Masks saved in: {p.RESULT_DIR}")
