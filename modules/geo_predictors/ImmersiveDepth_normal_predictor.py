import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image

from .geo_predictor import GeoPredictor
from .ImmersiveDepth.modules.unet import UNet
from .ImmersiveDepth.modules.midas.dpt_depth import DPTDepthModel
from .ImmersiveDepth.data.transforms import get_transform


# Define a class to predict normals using the DPTDepthModel model
class ImmersiveDepthNormalPredictor(GeoPredictor):
    def __init__(self):
        super().__init__()  # Initialize the superclass
        self.img_size = 384  # Define the target image size
        
        # Path to the pre-trained model checkpoint
        ckpt_path = './pre_checkpoints/omnidata_dpt_normal_v2.ckpt'
        
        # Initialize the DPTDepthModel with a specific backbone and number of channels
        self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
        
        # Move the model to CPU
        self.model.to(torch.device('cpu'))
        
        # Load the checkpoint and update the model weights
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            state_dict = {}
            # Remove the module prefix and update state dict
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        
        # Define transformations for input images
        self.trans_totensor = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
            transforms.CenterCrop(self.img_size),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        # Additional transformation set commented out, possibly for RGB images at different resolution
        # self.trans_rgb  = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
        #                                       transforms.CenterCrop(512)])

    def predict_normal(self, img):
        # Move the model to GPU for computation
        self.model.to(torch.device('cuda'))
        
        # Transform the input image to tensor and predict
        img_tensor = self.trans_totensor(img)
        output = self.model(img_tensor)
        
        # Move the model back to CPU
        self.model.to(torch.device('cpu'))
        
        # Additional interpolation commented out, for resizing output to a different resolution
        # output = F.interpolate(output[:, None], size=(512, 512), mode='bicubic')
        return output