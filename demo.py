import os
import sys
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch

# small hack to be able to import properly
# without modifying any files inside the cloned repositories
PROJECT_ROOT = Path(__file__).resolve().parent

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if len(sys.argv) < 2:
    image_path = r"images/000144_19.png"
else:
    image_path = sys.argv[1]
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# OpenCV reads BGR, the models expect RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# setting current working directory to deblurganv2
DEBLURGAN_ROOT = PROJECT_ROOT / "deblurganv2"
sys.path.insert(0, str(DEBLURGAN_ROOT))
os.chdir(DEBLURGAN_ROOT)

from deblurgan_adapter import DeblurGANv2Adapter

deblurGANv2Adapter = DeblurGANv2Adapter()
restored_dgan = deblurGANv2Adapter.deblur(image_rgb)

# setting current working directory to nafnet
NAFNET_ROOT = PROJECT_ROOT / "nafnet"
sys.path.insert(0, str(NAFNET_ROOT))
os.chdir(NAFNET_ROOT)

from nafnet_adapter import NAFNetAdapter

nafNetAdapter = NAFNetAdapter()
restored_nn = nafNetAdapter.deblur(image_rgb)

# setting current working directory to Restormer
RESTORMER_ROOT = PROJECT_ROOT / "Restormer"
sys.path.insert(0, str(RESTORMER_ROOT))
os.chdir(RESTORMER_ROOT)

from restormer_adapter import RestormerAdapter

restormerAdapter = RestormerAdapter()
restored_rs = restormerAdapter.deblur(image_rgb)

plt.figure(figsize = (12, 5))

plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(restored_dgan)
plt.title("Restored Image (DeblurGANv2)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(restored_nn)
plt.title("Restored Image (NAFNet)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(restored_rs)
plt.title("Restored Image (Restormer)")
plt.axis("off")

plt.show()