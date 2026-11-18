from runpy import run_path

import numpy as np
import torch
import torch.nn.functional as F

class RestormerAdapter:
    def __init__(self, weights_path = "Motion_Deblurring/pretrained_models/motion_deblurring.pth"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")

        parameters = {
            "inp_channels": 3,
            "out_channels": 3,
            "dim": 48,
            "num_blocks": [4, 6, 6, 8],
            "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8],
            "ffn_expansion_factor": 2.66,
            "bias": False,
            "LayerNorm_type": "WithBias",
            "dual_pixel_task": False
        }

        load_arch = run_path("basicsr/models/archs/restormer_arch.py")
        self.model = load_arch["Restormer"](**parameters)
        self.model.cuda()

        checkpoint = torch.load(
            weights_path
        )
        self.model.load_state_dict(checkpoint['params'])
        self.model.eval()

        self.img_multiple_of = 8

    def deblur(self, image):
        if image is None:
            raise ValueError("Image cannot be None!")

        image_tensor = (
            torch
            .from_numpy(image)
            .float()
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .cuda()
        )

        h, w = image_tensor.shape[2], image_tensor.shape[3]
        padH = (self.img_multiple_of - h % self.img_multiple_of) % self.img_multiple_of
        padW = (self.img_multiple_of - w % self.img_multiple_of) % self.img_multiple_of
        image_tensor = F.pad(
            image_tensor,
            (0, padW, 0, padH),
            "reflect"
        )

        with torch.inference_mode():
            restored = self.model(image_tensor)
        restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :h, :w]

        restored = (
            restored
            .permute(0, 2, 3, 1)
            .cpu()
            .detach()
            .numpy()[0] * 255.0
        ).astype(np.uint8)

        return restored
