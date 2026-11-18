import numpy as np
import torch
from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import img2tensor, tensor2img

class NAFNetAdapter:
    def __init__(self, config_path = "options/test/GoPro/NAFNet-width64.yml"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")

        opt = parse(
            config_path,
            is_train = False
        )

        opt["dist"] = False
        opt["num_gpu"] = torch.cuda.device_count()

        self.model = create_model(opt)

    def deblur(self, image):
        if image is None:
            raise ValueError("Image cannot be None!")

        # normalization
        image_float = image.astype(np.float32) / 255.0

        image_tensor = img2tensor(
            image_float,
            bgr2rgb = True,
            float32 = True
        )

        self.model.feed_data(data = {
            "lq": image_tensor.unsqueeze(0)
        })

        if self.model.opt["val"].get("grids", False):
            self.model.grids()

        self.model.test()

        if self.model.opt["val"].get("grids", False):
            self.model.grids_inverse()

        visuals = self.model.get_current_visuals()

        restored = tensor2img(
            [visuals["result"]]
        )

        return restored
        