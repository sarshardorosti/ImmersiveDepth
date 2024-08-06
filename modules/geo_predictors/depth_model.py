from MiDaS.midas.model_loader import load_model
import torch

class DepthModel:
    def __init__(self, model_weights):
        self.model = load_model(model_weights, model_type="omnidata_dpt_depth_v2")
        self.model.eval()

    def predict(self, img_tensor):
        with torch.no_grad():
            depth = self.model(img_tensor)
        return depth
