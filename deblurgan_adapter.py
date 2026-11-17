import torch

# manual preprocessing not necessary as
# DeblurGANv2 already preprocesses images
from predict import Predictor

class DeblurGANv2Adapter:
    def __init__(self, weights_path = None):
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            raise RuntimeError("CUDA is not available!")

        if weights_path is None:
            weights_path = "fpn_inception.h5"
        self.predictor = Predictor(
            weights_path = weights_path
        )

    def deblur(self, image):
        with torch.no_grad():
            restored = self.predictor(image, mask = None)

        if restored is None:
            raise RuntimeError("Predictor returned None!")

        return restored