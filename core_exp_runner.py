# Import required libraries and modules
import os
import cv2 as cv
import numpy as np
from shutil import copyfile
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig
from glob import glob
from tqdm import tqdm
import hydra

from modules.geo_predictors import PanoJointPredictor
from modules.dataset.dataset import WildDataset

from utils.utils import write_image, colorize_single_channel_image

# Define the DepthEstimator class to estimate depth from images
class DepthEstimator:
    def __init__(self, conf, device=torch.device('cuda')):
        """
        Initialize the DepthEstimator object.
        
        Parameters:
        conf (DictConfig): Configuration parameters from Hydra config files.
        device (torch.device): Device to run the calculations on (default: GPU if available).
        """
        self.conf = conf
        self.device = device

        # Load dataset as specified in the configuration
        self.dataset = WildDataset(conf.dataset)

        # Set up directories for saving outputs
        self.base_dir = os.getcwd()
        self.base_exp_dir = conf.device.base_exp_dir
        self.exp_dir = pjoin(self.base_exp_dir, '{}_{}'.format(conf['dataset_class_name'], self.dataset.case_name), conf.exp_name)

        # Ensure the experiment directory exists
        os.makedirs(self.exp_dir, exist_ok=True)

        # Initialize the geometric predictor for depth estimation
        self.geo_predictor = PanoJointPredictor()

    def estimate_depth(self):
        """
        Estimate depth from RGB panoramas using a trained model and save the output.
        """
        print('Estimating depth...')
        pano_rgb = self.dataset.image  # RGB image of the panorama
        ref_distance = self.dataset.ref_distance  # Reference distance map
        ref_normal = self.dataset.ref_normal  # Reference normal map

        # Visualization of reference distances
        write_image(pjoin(self.exp_dir, 'ImmersiveDepth_Distances.png'),
                    colorize_single_channel_image((ref_distance.min() + 1e-6) / (ref_distance + 1e-6)))

        # Visualization of normals if available
        if ref_normal is not None:
            write_image(pjoin(self.exp_dir, 'ImmersiveDepth_Normal.png'), (ref_normal * .5 + .5) * 255.)

        # Calculate and save new distances using the geometric predictor
        new_distances, _ = self.geo_predictor(pano_rgb, ref_distance, torch.ones_like(ref_distance))
        new_distances_np = new_distances.cpu().numpy().squeeze()
        new_distances_normalized = cv.normalize(new_distances_np, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        new_distances_path = pjoin(self.exp_dir, 'ImmersiveDepth_Depth.png')
        cv.imwrite(new_distances_path, new_distances_normalized)
        print(f"New distances saved to {new_distances_path}")
        
        # Indicate the end of the depth calculation process
        print("Depth calculation and saving completed. Stopping the project.")
        return

# Entry point for the script using Hydra to manage configuration
@hydra.main(version_base=None, config_path='./configs', config_name='nerf')
def main(conf: DictConfig) -> None:
    """
    Main function to initiate depth estimation.
    
    Args:
    conf (DictConfig): Configuration loaded from the Hydra framework.
    """
    # Set random seeds for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Initialize the DepthEstimator and start the depth estimation process
    runner = DepthEstimator(conf)
    runner.estimate_depth()

# Check if this script is the main program and run it
if __name__ == '__main__':
    main()