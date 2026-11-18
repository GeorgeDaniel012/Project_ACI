import torch

# manual preprocessing not necessary as
# DeblurGANv2 already preprocesses images
from predict import Predictor

class DeblurGANv2Adapter:
    def __init__(self, weights_path = "fpn_inception.h5"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")

        self.predictor = Predictor(
            weights_path = weights_path
        )

    def deblur(self, image):
        if image is None:
            raise ValueError("Image cannot be None!")

        with torch.no_grad():
            restored = self.predictor(image, mask = None)

        if restored is None:
            raise RuntimeError("Predictor returned None!")

        return restored