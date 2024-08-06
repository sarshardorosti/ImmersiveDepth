import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2 
import os
from PIL import Image 
import numpy as np

from .geo_predictor import GeoPredictor
from .depth_anything_v2.dpt import DepthAnythingV2
from .ImmersiveDepth_predictor import ImmersiveDepthPredictor, DepthAnything2
from .ImmersiveDepth_normal_predictor import ImmersiveDepthNormalPredictor

from modules.fields.networks import VanillaMLP
import tinycudann as tcnn

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import *

def scale_unit(x):
    """
    Normalize the input tensor `x` to the range [0, 1].
    
    Parameters:
    - x (torch.Tensor): The input tensor.

    Returns:
    - torch.Tensor: The normalized tensor.
    """
    return (x - x.min()) / (x.max() - x.min())



class SphereDistanceField(nn.Module):
    """
    Represents a sphere distance field using a neural network with a hash grid encoding.

    Attributes:
    - hash_grid (tcnn.Encoding): A hash grid encoding for input features.
    - geo_mlp (VanillaMLP): A multi-layer perceptron for predicting distances.

    Methods:
    - forward(directions, requires_grad=False): Computes distances and optionally gradients for given directions.
    """
    def __init__(self,
                 n_levels=16,
                 log2_hashmap_size=19,
                 base_res=16,
                 fine_res=2048):
        """
        Initialize the SphereDistanceField class with configurable parameters.

        Parameters:
        - n_levels (int): Number of levels in the hash grid.
        - log2_hashmap_size (int): Logarithmic size of the hash map.
        - base_res (int): Base resolution for the hash grid.
        - fine_res (int): Fine resolution for the hash grid.
        """
        super().__init__()
        
        # Calculate per level scale factor for hash grid resolution
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1))
        
        # Initialize hash grid encoding
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            }
        )

        # Initialize a vanilla MLP for geometric prediction
        self.geo_mlp = VanillaMLP(dim_in=n_levels * 2 + 3,
                                  dim_out=1,
                                  n_neurons=64,
                                  n_hidden_layers=2,
                                  sphere_init=True,
                                  weight_norm=False)

    def forward(self, directions, requires_grad=False):
        """
        Forward pass to compute distances and optionally gradients for given directions.

        Parameters:
        - directions (torch.Tensor): Input directions.
        - requires_grad (bool): If True, compute gradients.

        Returns:
        - torch.Tensor: Predicted distances.
        - Optional[torch.Tensor]: Predicted gradients if requires_grad is True.
        """
        if requires_grad:
            if not self.training:
                directions = directions.clone()  # get a copy to enable grad
            directions.requires_grad_(True)

        # Scale directions to [0, 1] range
        dir_scaled = directions * 0.49 + 0.49
        
        # Check if directions are within bounds
        selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32)
        
        # Encode the scaled directions using the hash grid
        scene_feat = self.hash_grid(dir_scaled)

        # Compute the distance using the MLP and apply softplus activation
        distance = F.softplus(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0] + 1.)

        # If gradient is required, compute it using autograd
        if requires_grad:
            grad = torch.autograd.grad(
                distance, directions, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance


class PanoJointPredictor(GeoPredictor):
    """
    A panoramic joint predictor that predicts depth and normals from images.

    Attributes:
    - depth_predictor1 (ImmersiveDepthPredictor): First depth predictor model.
    - depth_predictor2 (DepthAnything2): Second depth predictor model.
    - normal_predictor (ImmersiveDepthNormalPredictor): Normal predictor model.

    Methods:
    - grads_to_normal(grads): Converts gradients to normal vectors.
    - __call__(img, ref_distance, mask, ...): Performs depth inpainting on a single image.
    """
    def __init__(self):
        """
        Initialize the PanoJointPredictor class with depth and normal predictors.
        """
        super().__init__()
        self.depth_predictor1 = ImmersiveDepthPredictor()  # Initialize first depth predictor
        self.depth_predictor2 = DepthAnything2(encoder='vitl')  # Initialize second depth predictor with Vision Transformer Large encoder
        self.normal_predictor = ImmersiveDepthNormalPredictor()  # Initialize normal predictor

    def grads_to_normal(self, grads):
        """
        Convert gradients to normal vectors.

        Parameters:
        - grads (torch.Tensor): Input gradients with shape [H, W, 3].

        Returns:
        - torch.Tensor: Normal vectors with shape [H, W, 3].
        """
        height, width, _ = grads.shape
        
        # Convert image coordinates to panorama directions
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        
        # Generate orthogonal vectors
        ortho_a = torch.randn([height, width, 3])
        ortho_b = torch.linalg.cross(pano_dirs, ortho_a)
        ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
        ortho_a = torch.linalg.cross(ortho_b, pano_dirs)
        ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

        # Calculate val_a and val_b for normal estimation
        val_a = (grads * ortho_a).sum(-1, True) * pano_dirs + ortho_a
        val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
        val_b = (grads * ortho_b).sum(-1, True) * pano_dirs + ortho_b
        val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)

        # Calculate normals using cross product
        normals = torch.cross(val_a, val_b)
        normals = normals / torch.linalg.norm(normals, 2, -1, True)
        
        # Adjust normal vectors to ensure correct orientation
        is_inside = ((normals * pano_dirs).sum(-1, True) < 0.).float()
        normals = normals * is_inside + -normals * (1. - is_inside)
        return normals

    def __call__(self, img, ref_distance, mask, gen_res=384,
                 reg_loss_weight=1e-1, normal_loss_weight=1e-2, normal_tv_loss_weight=1e-2):
        """
        Perform depth inpainting on a single image.

        Parameters:
        - img (torch.Tensor): Input image with shape [H, W, 3].
        - ref_distance (torch.Tensor): Reference distance map with shape [H, W] or [H, W, 1].
        - mask (torch.Tensor): Mask indicating areas to inpaint with shape [H, W] or [H, W, 1].
        - gen_res (int): Resolution for generated perspective images.
        - reg_loss_weight (float): Regularization loss weight.
        - normal_loss_weight (float): Normal loss weight.
        - normal_tv_loss_weight (float): Normal total variation loss weight.

        Returns:
        - torch.Tensor: Inpainted distance map with shape [H, W].
        - torch.Tensor: Normal map with shape [H, W, 3].
        """
        # Prepare input tensors
        height, width, _ = img.shape
        device = img.device
        img = img.clone().squeeze().permute(2, 0, 1)  # Permute image dimensions to [3, H, W]
        mask = mask.clone().squeeze()[..., None].float().permute(2, 0, 1)  # Permute mask dimensions to [1, H, W]
        ref_distance = ref_distance.clone().squeeze()[..., None].float().permute(2, 0, 1)  # Permute reference distance dimensions to [1, H, W]
        ref_distance_mask = torch.cat([ref_distance, mask], 0)  # Concatenate reference distance and mask

        # Initialize lists to store perspective information
        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []

        # Generate perspective directions for different ratios
        for ratio in [1.1, 1.4, 1.7]:
            cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(
                gen_res=gen_res, ratio=ratio, ex_rot='rand')
            
            # Append current perspective directions and vectors to lists
            pers_dirs.append(cur_pers_dirs)
            pers_ratios.append(cur_pers_ratios)
            to_vecs.append(cur_to_vecs)
            down_vecs.append(cur_down_vecs)
            right_vecs.append(cur_right_vecs)

        # Concatenate perspective directions and vectors
        pers_dirs = torch.cat(pers_dirs, 0)
        pers_ratios = torch.cat(pers_ratios, 0)
        to_vecs = torch.cat(to_vecs, 0)
        down_vecs = torch.cat(down_vecs, 0)
        right_vecs = torch.cat(right_vecs, 0)

        # Calculate intrinsic camera parameters
        fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
        fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
        cx = torch.ones_like(fx) * gen_res * .5
        cy = torch.ones_like(fy) * gen_res * .5

        # Move perspective directions and vectors to device
        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        # Calculate rotation matrices for world-to-camera and camera-to-world transformations
        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)
        rot_c2w = torch.linalg.inv(rot_w2c)

        # Number of perspective directions
        n_pers = len(pers_dirs)
        
        # Convert perspective directions to image coordinates and sample coordinates
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        # Sample perspective images using grid sampling
        pers_imgs = F.grid_sample(img[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border')  # [n_pers, 3, gen_res, gen_res]
        
        # Initialize lists to store predicted distances and normals
        pred_distances_raw = []
        pred_normals_raw = []

        pred_distances_1 = []
        pred_distances_2 = []

        # Iterate over perspective images to predict distances and normals
        for i in range(n_pers):
            with torch.no_grad():
                # Define intrinsic parameters for current perspective
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }
                
                # Predict depth using the first depth predictor
                pred_depth1 = self.depth_predictor1.predict_depth1(pers_imgs[i: i+1], intri=intri).clip(0., None)  # [1, 1, 384, 384]
                pred_depth1 = pred_depth1 / (pred_depth1.mean() + 1e-5)
                pred_distances_1.append(pred_depth1 * pers_ratios[i].permute(2, 0, 1)[None])

                # Predict depth using the second depth predictor (DepthAnything2)
                image = pers_imgs[i: i+1]  # Select a single perspective image
                img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to numpy array
                
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))  # Convert numpy array to PIL Image
                
                # Predict depth using the DepthAnything2 model
                pred_depth2 = self.depth_predictor2.predict_depth2(img_pil).clip(0., None)

                pred_distances_2.append(pred_depth2[:, :, :384, :384] * pers_ratios[i].permute(2, 0, 1)[None])  # Clip to original dimensions

                # Predict normals using the normal predictor
                pred_normals = self.normal_predictor.predict_normal(pers_imgs[i: i+1])
                pred_normals = pred_normals * 2. - 1.
                pred_normals = pred_normals / torch.linalg.norm(pred_normals, ord=2, dim=1, keepdim=True)

                # Apply camera-to-world rotation to predicted normals
                pred_normals = pred_normals.permute(0, 2, 3, 1)  # [1, res, res, 3]
                pred_normals = apply_rot(pred_normals, rot_c2w[i])
                pred_normals = pred_normals.permute(0, 3, 1, 2)  # [1, 3, res, res]
                pred_normals_raw.append(pred_normals)

        # Concatenate predicted distances
        pred_distances_1 = torch.cat(pred_distances_1, dim=0)  # [n_pers, 1, 384, 384]
        pred_distances_2 = torch.cat(pred_distances_2, dim=0)  # [n_pers, 1, 384, 384]
        
        # Compute weighted average of predicted distances from both models
        weight1 = 1.00  # Weight for the first model
        weight2 = 0.40  # Weight for the second model
        pred_distances_raw = weight1 * pred_distances_1 + weight2 * pred_distances_2  # [n_pers, 1, 384, 384]

        pred_normals_raw = torch.cat(pred_normals_raw, dim=0)  # [n_pers, 3, res, res]
        pers_dirs = pers_dirs.permute(0, 3, 1, 2)

        # Concatenate perspective directions, distances, and normals
        sup_infos = torch.cat([pers_dirs, pred_distances_raw, pred_normals_raw], dim=1)

        # Initialize parameters for scale, bias, and sphere distance field
        scale_params = torch.zeros([n_pers], requires_grad=True)
        bias_params_global = torch.zeros([n_pers], requires_grad=True)
        bias_params_local_distance = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True)
        bias_params_local_normal = torch.zeros([n_pers, 3, 128, 128], requires_grad=True)

        sp_dis_field = SphereDistanceField()

        # Stage 1: Optimize global parameters
        all_iter_steps = 1500
        lr_alpha = 1e-2
        init_lr = 1e-1
        init_lr_sp = 1e-2
        init_lr_local = 1e-1
        local_batch_size = 256

        # Initialize optimizers for global, local, and sphere distance field parameters
        optimizer_sp = torch.optim.Adam(sp_dis_field.parameters(), lr=init_lr_sp)
        optimizer_global = torch.optim.Adam([scale_params, bias_params_global], lr=init_lr)
        optimizer_local = torch.optim.Adam([bias_params_local_distance, bias_params_local_normal], lr=init_lr_local)

        # Perform optimization in two phases: global and hybrid
        for phase in ['global', 'hybrid']:
            for iter_step in tqdm(range(all_iter_steps)):
                # Calculate progress and adjust learning rate
                progress = iter_step / all_iter_steps
                if phase == 'global':
                    progress = progress * .5
                else:
                    progress = progress * .5 + .5

                lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                for g in optimizer_global.param_groups:
                    g['lr'] = init_lr * lr_ratio
                for g in optimizer_local.param_groups:
                    g['lr'] = init_lr_local * lr_ratio
                for g in optimizer_sp.param_groups:
                    g['lr'] = init_lr_sp * lr_ratio

                # Randomly select a perspective index
                idx = np.random.randint(low=0, high=n_pers)
                
                # Sample local batch coordinates
                sample_coords = torch.rand(n_pers, local_batch_size, 1, 2) * 2. - 1
                
                # Sample support information, distance bias, and normal bias
                cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')  # [n_pers, 7, local_batch_size, 1]
                distance_bias = F.grid_sample(bias_params_local_distance, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)
                normal_bias = F.grid_sample(bias_params_local_normal, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                normal_bias = normal_bias[:, :, :, 0].permute(0, 2, 1)

                # Extract direction, reference predicted distances, and normals
                dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)  # [n_pers, local_batch_size, 3]
                dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)

                ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)  # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances * F.softplus(scale_params[:, None, None])  # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances + distance_bias

                ref_normals = cur_sup_info[:, 4:, :, 0].permute(0, 2, 1)
                ref_normals = ref_normals + normal_bias
                ref_normals = ref_normals / torch.linalg.norm(ref_normals, 2, -1, True)

                # Compute predicted distances and gradients
                pred_distances, pred_grads = sp_dis_field(dirs.reshape(-1, 3), requires_grad=True)
                pred_distances = pred_distances.reshape(n_pers, local_batch_size, 1)
                pred_grads = pred_grads.reshape(n_pers, local_batch_size, 3)

                # Calculate distance loss using Smooth L1 loss
                distance_loss = F.smooth_l1_loss(ref_pred_distances, pred_distances, beta=5e-1, reduction='mean')

                # Generate orthogonal vectors for normal calculation
                ortho_a = torch.randn([n_pers, local_batch_size, 3])
                ortho_b = torch.linalg.cross(dirs, ortho_a)
                ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
                ortho_a = torch.linalg.cross(ortho_b, dirs)
                ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

                # Calculate val_a and val_b for normal estimation
                val_a = (pred_grads * ortho_a).sum(-1, True) * dirs + ortho_a
                val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
                val_b = (pred_grads * ortho_b).sum(-1, True) * dirs + ortho_b
                val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)
                
                # Calculate errors in normal estimation
                error_a = (val_a * ref_normals).sum(-1, True)
                error_b = (val_b * ref_normals).sum(-1, True)
                errors = torch.cat([error_a, error_b], -1)
                
                # Calculate normal loss using Smooth L1 loss
                normal_loss = F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=5e-1, reduction='mean')

                # Calculate regularization loss for scale parameters
                reg_loss = ((F.softplus(scale_params).mean() - 1.)**2).mean()

                # Calculate total variation loss for local bias if in hybrid phase
                if phase == 'hybrid':
                    distance_bias_local = bias_params_local_distance
                    distance_bias_tv_loss = F.smooth_l1_loss(distance_bias_local[:, :, 1:, :], distance_bias_local[:, :, :-1, :], beta=1e-2) + \
                                            F.smooth_l1_loss(distance_bias_local[:, :, :, 1:], distance_bias_local[:, :, :, :-1], beta=1e-2)
                    normal_bias_local = bias_params_local_normal
                    normal_bias_tv_loss = F.smooth_l1_loss(normal_bias_local[:, :, 1:, :], normal_bias_local[:, :, :-1, :], beta=1e-2) + \
                                          F.smooth_l1_loss(normal_bias_local[:, :, :, 1:], normal_bias_local[:, :, :, :-1], beta=1e-2)

                else:
                    distance_bias_tv_loss = 0.
                    normal_bias_tv_loss = 0.

                # Sample reference distance and mask
                pano_image_coords = direction_to_img_coord(dirs.reshape(-1, 3))
                pano_sample_coords = img_coord_to_sample_coord(pano_image_coords)  # [all_batch_size, 2]
                sampled_ref_distance_mask = F.grid_sample(ref_distance_mask[None], pano_sample_coords[None, :, None, :], padding_mode='border')  # [1, 2, batch_size, 1]
                sampled_ref_distance = sampled_ref_distance_mask[0, 0]
                sampled_ref_mask = sampled_ref_distance_mask[0, 1]
                
                # Calculate reference distance loss
                ref_distance_loss = F.smooth_l1_loss(sampled_ref_distance.reshape(-1), pred_distances.reshape(-1), beta=1e-2, reduction='none')
                ref_distance_loss = (ref_distance_loss * (sampled_ref_mask < .5).reshape(-1)).mean()

                # Calculate total loss
                loss = ref_distance_loss * 20. * progress + \
                       distance_loss + reg_loss * reg_loss_weight +\
                       normal_loss * normal_loss_weight +\
                       distance_bias_tv_loss * 1. +\
                       normal_bias_tv_loss * normal_tv_loss_weight

                # Zero gradients and perform backpropagation
                optimizer_global.zero_grad()
                optimizer_sp.zero_grad()
                if phase == 'hybrid':
                    optimizer_local.zero_grad()

                loss.backward()
                
                # Update parameters
                optimizer_global.step()
                optimizer_sp.step()
                if phase == 'hybrid':
                    optimizer_local.step()

        # Get new distance map and normal map
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        new_distances, new_grads = sp_dis_field(pano_dirs.reshape(-1, 3), requires_grad=True)
        new_distances = new_distances.detach().reshape(height, width, 1)
        new_normals = self.grads_to_normal(new_grads.detach().reshape(height, width, 3))

        return new_distances, new_normals