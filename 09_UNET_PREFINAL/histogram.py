import parameter as p
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# Función de ecualización de histograma
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')

    img_eq = cdf_final[image]
    return img_eq.astype(np.float32) / 255.0

# Cargar una imagen de ejemplo en escala de grises
img_path = os.path.join(p.INFERENCE_PATH, "ID00009637202177434476278_75.jpg")
img = Image.open(img_path).convert("L")
img_np = np.array(img, dtype=np.float32) / 255.0

# Aplicar ecualización
img_eq = histogram_equalization(img_np)

# Mostrar resultados
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(img_np, cmap="gray")
axes[0, 0].set_title("Imagen original")
axes[0, 0].axis("off")

axes[0, 1].hist(img_np.flatten(), bins=50, color="gray")
axes[0, 1].set_title("Histograma original")

axes[1, 0].imshow(img_eq, cmap="gray")
axes[1, 0].set_title("Imagen ecualizada")
axes[1, 0].axis("off")

axes[1, 1].hist(img_eq.flatten(), bins=50, color="gray")
axes[1, 1].set_title("Histograma ecualizado")

plt.tight_layout()
output_path = os.path.join(p.RESULT_DIR, "hist_eq_result.png")
plt.savefig(output_path)
plt.close()

print(f"✅ Gráfico guardado como '{output_path}'")
