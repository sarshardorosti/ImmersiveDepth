# ImmersiveDepth: Advanced 360-Degree Depth Estimation

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Totoro97/PeRF)

<div align="center">
    <img src="MY---------------------" width="90%"/>
</div>

## Overview

ImmersiveDepth is a cutting-edge project designed to provide robust and accurate monocular depth estimation (MDE) from 360-degree images. By integrating advanced models such as MiDaS v3.1 and Depth Anything V2, this project addresses the challenges of depth perception under varying environmental conditions, scale variations, and image complexities. It leverages the strengths of both models to enhance overall performance, offering a comprehensive solution for applications in Animation, Game, AR, VR, and 3D visualization.

## Features

- **Model Integration**: Combines MiDaS v3.1 and Depth Anything V2 to leverage the strengths of both models for improved depth prediction accuracy.
- **Tangent Image Projection**: Uses an icosahedron-based approach to project spherical images into perspective tangent images, allowing for precise depth map creation.
- **Global Disparity Alignment**: Aligns predicted disparity maps using advanced affine transformation techniques to ensure consistent global alignment.
- **Robust Preprocessing**: Employs a range of image enhancement techniques such as Gaussian filtering, CLAHE, and dehazing to improve depth map quality.
- **Efficient Workflow**: Optimized for GPU processing with minimal hardware requirements, providing faster and more efficient depth estimation.

## Table of Contents

- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [References](#references)

## Project-Structure
```
ImmersiveDepth
├── 3D
├── configs
│   └── device
├── modules
│   ├── dataset
│   ├── fields
│   └── geo_predictors
│       └── ImmersiveDepth
│           ├── config
│           │   ├── depth.yaml
│           │   └── normal.yaml
│           ├── data
│           │   └── splits
│           ├── ImmersiveDepth_dataset.py
│           ├── augmentation.py
│           ├── masks.py
│           ├── refocus_augmentation.py
│           └── splits.py
│       ├── dataloader
│       │   ├── component_datasets
│       ├── losses
│       ├── modules
│       │   ├── midas
│       │   └── unet.py
│       ├── tools
│       └── depth_anything_v2
│           ├── dinov2_layers
│           └── util
│               └── pre_checkpoints
│               ├── utils
│                   └── requirements.txt
```


## Methodology

### 1. Initialization and Configuration

The initialization phase involves setting up the environment, loading configuration files, and preparing the system for depth estimation. Key files in this stage include:

- **`core_exp_runner.py`**: This script uses `Hydra` to load configurations from `config.yaml`, `depth.yaml`, and `normal.yaml`, setting parameters such as model paths and data directories. It initializes the `DepthEstimator` class, which manages the entire estimation process.

    ```python
    @hydra.main(config_path=".", config_name="config")
    def main(cfg: DictConfig):
        set_random_seed(cfg.get("seed", 0))
        depth_estimator = DepthEstimator(cfg)
        depth_estimator.estimate_depth()
    ```

- **`config.yaml`**: This file contains global settings for the project, including paths to models and datasets, as well as various processing options.

### 2. Data Loading and Preprocessing

This stage handles the loading of datasets and initial preprocessing of images, including resizing and normalization.

- **`dataset.py`**: Manages dataset loading and

 preprocessing. The `WildDataset` class reads panoramic images and associated reference data, performing resizing and normalization to prepare data for model input.

    ```python
    class WildDataset(Dataset):
        def __init__(self, image_path, image_resize, ref_dist_map=None, ref_normal_map=None):
            self.image_path = image_path
            self.image_resize = image_resize
            self.images = self._load_images()
            self.ref_dist_map = ref_dist_map
            self.ref_normal_map = ref_normal_map

        def _load_images(self):
            # Load images from the specified path
            images = []
            for img_file in sorted(os.listdir(self.image_path)):
                img_path = os.path.join(self.image_path, img_file)
                img = cv2.imread(img_path)
                images.append(img)
            return images
    ```

- **Data Augmentation**: Implemented in files like `transforms.py`, `augmentation.py`, and `refocus_augmentation.py`, these modules apply transformations such as resizing, normalization, and refocusing to enhance data diversity and model robustness.

    ```python
    class Resize(object):
        def __call__(self, sample):
            width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
            sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)
            return sample

    class NormalizeImage(object):
        def __call__(self, sample):
            sample["image"] = (sample["image"] - self.__mean) / self.__std
            return sample
    ```

### 3. Model Architecture and Processing

This phase leverages advanced model architectures for depth estimation, using Vision Transformers and multi-scale feature fusion.

- **Depth Prediction Models**: Files such as `dpt_depth.py`, `midas_net.py`, `midas_net_custom.py`, and `depth_model.py` implement models like `DPTDepthModel` and `MidasNet`, utilizing Vision Transformers for feature extraction and custom decoders for depth map generation.

- **Vision Transformer Components**: Implemented in `vit.py`, these components transform image patches into sequences of tokens processed through transformer layers.

- **Depth Anything V2 Model Components**: Implemented in the `depth_anything_v2/` directory, this module includes components like `attention.py`, `block.py`, `layer_scale.py`, and others, providing a robust architecture for depth estimation.

### 4. Prediction and Refinement

In this stage, the system predicts depth and normal maps, refining them for geometric consistency and combines outputs from multiple models for enhanced results.

#### Combination of Model Outputs

The combination of outputs from **MiDaS v3.1** and **Depth Anything V2** is crucial for improving depth prediction accuracy. This is achieved by leveraging the strengths of each model to mitigate their respective weaknesses. The integration of these models is primarily handled in:

- **`ImmersiveDepth_predictor.py`** and **`ImmersiveDepth_normal_predictor.py`**: These files handle the prediction and integration of depth and normal maps from both models, combining their strengths to improve the final output.

- **`pano_joint_predictor.py`**: This file is key for integrating the models, as it combines the predictions from both depth and normal map models to produce better results.

    ```python
    class PanoJointPredictor(GeoPredictor):
        def __init__(self):
            super().__init__()
            self.depth_predictor1 = MidasPredictor()  # Initialize first depth predictor
            self.depth_predictor2 = DepthAnything2(encoder='vitl')  # Initialize second depth predictor with Vision Transformer Large encoder
            self.normal_predictor = OmniDepthNormalPredictor()  # Initialize normal predictor

        def __call__(self, img, ref_distance, mask, gen_res=384, ...):
            for i in range(n_pers):
                with torch.no_grad():
                    intri = {'fx': fx[i].item(), 'fy': fy[i].item(), 'cx': cx[i].item(), 'cy': cy[i].item()}
                    pred_depth1 = self.depth_predictor1.predict_depth1(pers_imgs[i: i+1], intri=intri).clip(0., None)
                    pred_depth2 = self.depth_predictor2.predict_depth2(img_pil).clip(0., None)
                    # Combine outputs from both models
                    pred_depth_combined = self.combine_depths(pred_depth1, pred_depth2)
                    pred_normals = self.normal_predictor.predict_normal(pers_imgs[i: i+1])
    ```
#### Image Post-Processing

Image post-processing is applied to enhance the quality of the output depth maps, addressing issues like noise, fog, and sharpness. This is accomplished through various techniques implemented in:

- **`augmentation.py`**: This file applies image enhancement techniques such as Gaussian filtering, CLAHE (Contrast Limited Adaptive Histogram Equalization), and dehazing to improve the clarity and quality of the depth maps.

    ```python
    class GaussianFilter(object):
        def __call__(self, image):
            return cv2.GaussianBlur(image, (5, 5), 0)

    class CLAHE(object):
        def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        def __call__(self, image):
            return self.clahe.apply(image)

    class Dehazing(object):
        def __call__(self, image):
            # Implement dehazing technique
            ...

    # Applying these augmentations in the preprocessing pipeline
    class AugmentationPipeline:
        def __init__(self):
            self.augmentations = [GaussianFilter(), CLAHE(), Dehazing()]

        def __call__(self, image):
            for augmentation in self.augmentations:
                image = augmentation(image)
            return image
    ```
The combination of model outputs and image post-processing plays a critical role in the ImmersiveDepth project. By intelligently integrating predictions from multiple models and applying advanced image processing techniques, the system achieves higher accuracy and stability in depth estimation. This approach significantly improves the system's resilience to various environmental conditions and enhances detail recognition, providing robust depth estimation solutions.

---
- **Geometry Refinement**: Implemented in `pano_geo_refiner.py` and `geo_utils.py`, this process refines depth and normal maps by optimizing geometric consistency, aligning predictions across multiple views, and iteratively improving results.

    ```python
    class PanoGeoRefiner:
        def refine(self, distances, normals):
            # Iteratively refine depth and normal maps for accuracy
            ...

    def align_scale(data, reference):
        # Aligns scales between data and reference for consistency
        ...
    ```

### 5. Mask Generation and Instance Segmentation

This step generates masks to indicate valid geometry regions and performs instance segmentation to extract and visualize object instances from images.

- **Mask Generation**: Implemented in `masks.py`, this module generates masks that identify valid geometry within the image, focusing models on important areas during prediction.

- **Instance Segmentation**: Implemented in `segment_instance.py`, this module extracts object instances using techniques like bounding box extraction and mask application, supporting tasks like semantic segmentation and object recognition.


### 6. Visualization and Output Handling

This stage involves visualizing and saving predicted depth maps, as well as generating 3D models for further analysis.

- **Visualization Utilities**: Implemented in `utils.py` and `viz_utils.py`, these modules visualize depth and normal maps, providing graphical representations of the results.

- **Output Management**: Uses utilities like `write_image` to save results, enabling detailed evaluation and analysis.

#### Roles of Key Files

- **`core_exp_runner.py`**: Executes the depth estimation workflow, using `DepthModel` and `GeoPredictor` to predict and save depth maps as PLY files.

- **`3dopen.py`**: Loads and visualizes 3D models using Open3D, allowing for interactive examination of PLY files.

- **`exportmesh.py`**: Converts PLY files to FBX format, making 3D models compatible with various 3D applications.

These scripts work together to ensure that the project delivers robust and accurate 3D depth estimation and visualization capabilities, suitable for applications such as VR, AR, and 3D visualization.

---

## Installation

### Step 1: Clone the Repository

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/sarshardorosti/ImmersiveDepth.git
cd ImmersiveDepth
pip install -r requirements.txt
```

**Recommended Setup**: Python 3.9, CUDA 12.1, and PyTorch 2.3.

---

### Step 2: Download Pre-trained Checkpoints

To achieve accurate depth and normal estimations, you need to download the pre-trained model checkpoints. Follow the instructions below to set up the models correctly.

#### Primary Depth and Normal Detection Models

Download the primary depth and normal detection models from the link below and place them in the `pre_checkpoints` directory:

- [Download Primary Models](https://github.com/sarshardorosti/ImmersiveDepth/tree/main/pre_checkpoints)

   ```plaintext
   ImmersiveDepth/
   └── pre_checkpoints/
       ├── model_primary_depth.pth
       ├── model_primary_normal.pth
       └── ...
   ```

#### Secondary Depth Detection Model

Download the secondary depth detection model from the link below and place it in the `modules/geo_predictors/checkpoints` directory:

- [Download Secondary Model](https://github.com/sarshardorosti/ImmersiveDepth/tree/main/modules/geo_predictors/checkpoints)

   ```plaintext
   ImmersiveDepth/
   └── modules/
       └── geo_predictors/
           └── checkpoints/
               ├── model_secondary_depth.pth
               └── ...
   ```

Ensure that all the model files are correctly placed in their respective directories before proceeding with the usage and training sections.

---

## Usage

### Training and Evaluation

#### Train on Example Data

Use the following command to train the model on example data:

```bash
python core_exp_runner.py --config-name nerf dataset.image_path=$(pwd)/example_data/hotel/hotel.png device.base_exp_dir=$(pwd)/exp
```

This command initializes the training process, loading the example dataset and preparing the system for depth estimation.


#### Open 3D Model

To visualize and interact with the generated 3D models, run the following command:

```bash
python core_exp_runner.py --config-name nerf dataset.image_path=$(pwd)/example_data/kitchen/image.png device.base_exp_dir=$(pwd)/exp mode=render_dense is_continue=true
```



## Evaluation Metrics

The project's performance is evaluated using metrics such as Absolute Relative Error (AbsRel), Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and more. These metrics ensure that depth predictions are accurate and reliable, aligning closely with ground-truth values.

- **Absolute Relative Error (AbsRel)**: Measures the average of the absolute differences between the predicted depth and the ground-truth depth, relative to the ground-truth depth.
- **Mean Absolute Error (MAE)**: Calculates the average of the absolute differences between the predicted depth and the ground-truth depth.
- **Root Mean Square Error (RMSE)**: Measures the square root of the average of the squared differences between the predicted depth and the ground-truth depth.
- **Log Root Mean Square Error (RMSE(log))**: Similar to RMSE but operates in the logarithmic space, measuring the square root of the average of the squared logarithmic differences.
- **Square Relative Error (SqRel)**: Calculates the average of the squared differences between the predicted depth and the ground-truth depth, relative to the ground-truth depth.


## Results

The developed system has demonstrated superior performance in creating precise distance maps from 360-degree images. By combining outputs from both MiDaS v3.1 and Depth Anything V2, the project achieves high accuracy and stability in depth predictions. The results indicate significant improvements over existing models, with enhanced detail recognition and resilience to environmental variations.


### Key Findings

- **Improved Accuracy**: The integrated model outperforms standalone models in various metrics, indicating enhanced depth prediction accuracy.
- **Efficient Processing**: Optimized for GPU processing, the system provides efficient depth estimation with reduced hardware requirements.
- **Comprehensive Solution**: The project offers a robust solution for depth estimation, applicable in VR, 3D mapping, and other domains.


## References

- Ainaz Eftekhar, Sax, A., Malik, J., & Zamir, A. (2021). Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans. 2021 IEEE/CVF International Conference on Computer Vision (ICCV). [arxiv.org](https://arxiv.org/abs/2110.04994)
- Birkl, R., Wofk, D., & Müller, M. (2023). MiDaS v3.1 -- A Model Zoo for Robust Monocular Relative Depth Estimation. [arxiv.org](https://arxiv.org/abs/2307.14460)
- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2022). Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence. [arxiv.org](https://arxiv.org/pdf/1907.01341v3.pdf)
- Rey-Area, M., Yuan, M., & Richardt, C. (2022). 360MonoDepth: High-Resolution 360° Monocular Depth Estimation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- Wang, G., Wang, P., Chen, Z., Wang, W., Loy, C. C., & Liu, Z. (2023). PERF: Panoramic Neural Radiance Field from a Single Panorama. [arxiv.org](https://arxiv.org/abs/2310.16831v2)
- Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything V2. [arxiv.org](https://arxiv.org/abs/2406.09414)




