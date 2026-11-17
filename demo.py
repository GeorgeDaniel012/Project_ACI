import os
import sys
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np

# small hack to be able to import properly
# without modifying any files inside the cloned repositories
PROJECT_ROOT = Path(__file__).resolve().parent

if len(sys.argv) < 2:
    image_path = r'images/images.png'
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
restored = deblurGANv2Adapter.deblur(image_rgb)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.clip(restored, 0, 1))
plt.title("Restored Image (DeblurGANv2)")
plt.axis("off")

plt.imshow(restored)

plt.show()