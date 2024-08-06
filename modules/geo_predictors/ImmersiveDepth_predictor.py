import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2 
import os
import PIL
from PIL import Image
import numpy as np

from .geo_predictor import GeoPredictor
from .depth_anything_v2.dpt import DepthAnythingV2
from .ImmersiveDepth.modules.unet import UNet
from .ImmersiveDepth.modules.midas.dpt_depth import DPTDepthModel
from .ImmersiveDepth.data.transforms import get_transform


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    # Apply mask if provided to set invalid areas to NaN
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    
    # Flatten the image and sort the pixel values to remove NaN values for processing
    sorted_img = torch.sort(torch.flatten(img))[0]
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]  # Remove NaNs

    # Truncate outliers based on a percentage value to focus on the core distribution
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6  # Epsilon to prevent division by zero in normalization
    
    # Replace NaNs with the mean and standardize the image using calculated mean and variance
    img = torch.nan_to_num(img, nan=trunc_mean.item())
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img

# Define a class for depth prediction using a specific type of neural network
class ImmersiveDepthPredictor(GeoPredictor):
    def __init__(self):
        super().__init__()
        self.img_size = 384  # Image size for processing
        ckpt_path = './pre_checkpoints/omnidata_dpt_depth_v2.ckpt'
        
        # Initialize the DPTDepthModel with a specific configuration
        self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=1)
        self.model.to(torch.device('cpu'))
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        
        # Load the model weights
        if 'state_dict' in checkpoint:
            state_dict = {k[6:]: v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        
        # Define transforms for input image preprocessing
        self.trans_totensor = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
            transforms.CenterCrop(self.img_size),
            transforms.Normalize(mean=0.5, std=0.5)])
        
    def predict_disparity(self, img, **kwargs):
        # Switch model to CUDA, process the image, perform inference, and process output
        self.model.to(torch.device('cuda'))
        img_tensor = self.trans_totensor(img)
        output = self.model(img_tensor).clip(0., 1.)
        self.model.to(torch.device('cpu'))
        output = 1. / (output + 1e-6)
        return output[:, None]

    # Similar function as `predict_disparity` but may have different intended adjustments or post-processing
    def predict_depth1(self, img, **kwargs):
        self.model.to(torch.device('cuda'))
        img_tensor = self.trans_totensor(img)
        output = self.model(img_tensor).clip(0., 1.)
        self.model.to(torch.device('cpu'))
        return output[:, None]

# Define another depth prediction class for different model configurations
class DepthAnything2:
    def __init__(self, encoder='vitl'):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Configuration for the depth model based on encoder type
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        # Load model and weights
        self.depth_anything = DepthAnythingV2(**self.model_configs[encoder])
        self.depth_anything.load_state_dict(torch.load('/home/s5639776/MasterProject/test/ImmersiveDepth/modules/geo_predictors/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.DEVICE).eval()

    # Function to apply a texture effect on an image
    def apply_texture(self, image, amount):
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        high_pass = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return high_pass

    # Function to enhance the clarity of an image
    def apply_clarity(self, image, amount):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=amount, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final

    # Function to reduce haze in an image
    def apply_dehaze(self, image, amount):
        return cv2.detailEnhance(image, sigma_s=20, sigma_r=0.03 * amount)

    # Use the DepthAnythingV2 model to predict depth, apply effects and normalize
    def predict_depth2(self, image_pil, input_size=384):
        image_np = np.array(image_pil)
        depth = self.depth_anything.infer_image(image_np, input_size)
        depth = 1.0 - (depth / depth.max())  # Normalize and invert
        depth_rgb = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Apply texture, clarity, and dehaze effects sequentially
        texture_amount, clarity_amount, dehaze_amount = 0.3, 2.2, 0.8
        texture_image = self.apply_texture(depth_rgb, texture_amount)
        clarity_image = self.apply_clarity(texture_image, clarity_amount)
        dehazed_image = self.apply_dehaze(clarity_image, dehaze_amount)
        
        depth_gray = cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2GRAY)
        depth_normalized = depth_gray / 255.0
        pred_depth2 = torch.tensor(depth_normalized).unsqueeze(0).unsqueeze(0).float().to(self.DEVICE)
        return pred_depth2

# Command-line interface for using the depth prediction functionality
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--input-matrix', type=str, required=True, help='Path to the input matrix file (numpy .npy file)')
    parser.add_argument('--input-size', type=int, default=384)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    args = parser.parse_args()
    
    image_matrix = np.load(args.input_matrix)
    depth_model = DepthAnything2(encoder=args.encoder)
    depth_matrix = depth_model.predict_depth2(image_matrix, args.input_size)
    
    print(depth_matrix)